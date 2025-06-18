import torch
from gemmi import UnitCell
from torch_geometric.data import Data

from elements import element_dict


def featurize(
    frac_coords: torch.Tensor,
    cell: UnitCell,
    atom_symbols: list[str],
    ap_dict: dict,
):
    # prepare chemical data
    atom_nums = torch.tensor([element_dict[k] for k in atom_symbols], dtype=torch.long)
    ap_tensor = ap_dict["properties"]
    atom_features = ap_tensor[atom_nums]
    en_index = ap_dict["en_index"]
    en_values = atom_features[:, en_index]
    bonds_mat = ap_dict["bonds"]

    # parameters for descriptors
    r_0 = 1.0
    rad_cut = 8.0
    ang_cut = 6.0
    rdf_binsize = 0.25
    rdf_alpha = 60.0
    racsf_binsize = 0.5
    racsf_eta = 6.0
    adf_binsize_deg = 10.0
    adf_beta = 60.0
    aacsf_n_eta = 8

    # create super cells and Cartesian coordinates
    max_cart = torch.tensor(max((rad_cut, ang_cut)), dtype=torch.float).repeat(3)
    frac_mat = torch.tensor(cell.frac.mat.tolist(), dtype=torch.float)
    super_scale = (frac_mat @ max_cart).abs().ceil()
    list_iters = [torch.arange(i, dtype=torch.float) for i in super_scale * 2 + 1]
    super_opts = torch.cartesian_prod(*list_iters) - super_scale
    nso = len(super_opts)
    frac_tiles = torch.tile(frac_coords, (nso, 1, 1))
    super_tiles = torch.repeat_interleave(super_opts, len(frac_coords), dim=0)
    orth_mat = torch.tensor(cell.orth.mat.tolist(), dtype=torch.float)
    cart_super = (frac_tiles - super_tiles.view(frac_tiles.shape)) @ orth_mat
    cart_unit = cart_super[nso // 2]

    # inits for atomic property weighted radial distribution functions (wRDF)
    rdf_nbins = int((rad_cut - r_0) / rdf_binsize)
    rdf_init = r_0 + rdf_binsize
    rdf_edges = torch.linspace(rdf_init, rad_cut, rdf_nbins, dtype=torch.float)

    # inits for radial ACSF
    racsf_nbins = int((rad_cut - r_0) / racsf_binsize)
    racsf_init = r_0 + racsf_binsize
    racsf_edges = torch.linspace(racsf_init, rad_cut, racsf_nbins, dtype=torch.float)

    # inits for atomic property weighted angular distribution functions (wADF)
    adf_binsize = torch.tensor(adf_binsize_deg, dtype=torch.float).deg2rad()
    adf_nbins = (torch.pi / adf_binsize).long()
    adf_edges = torch.linspace(adf_binsize, torch.pi, adf_nbins, dtype=torch.float)

    # inits for angular ACSF
    aacsf_nbins = aacsf_n_eta * 2
    aacsf_edges = torch.linspace(r_0, ang_cut, aacsf_n_eta, dtype=torch.float)
    aacsf_eta = 1 / (2 * (aacsf_edges**2))
    aacsf_lambda = torch.tensor((-1.0, 1.0), dtype=torch.float)
    aacsf_lambda_bins, aacsf_eta_bins = torch.cartesian_prod(aacsf_lambda, aacsf_eta).T

    # generate descriptors for one atom at a time to save memory
    features_list = []
    bond_pairs_list = []
    bond_dists_list = []
    for i, cart_i in enumerate(cart_unit):
        # compute distances between atom i to all other atoms in the supercell
        dist_to_i = torch.linalg.norm(cart_super - cart_i, dim=-1)

        # obtain bonds
        bonds_super = dist_to_i < bonds_mat[atom_nums, atom_nums[i]]
        bonds_super &= dist_to_i != 0.0
        bonds_j = bonds_super.nonzero()[:, 1]
        bond_pairs_list.append(torch.vstack([torch.full_like(bonds_j, i), bonds_j]))
        bond_dists_list.append(dist_to_i[bonds_super])

        # obtain distances within the cutoffs for radial descriptors
        rad_bool = (dist_to_i >= r_0) & (dist_to_i <= rad_cut)
        rad_p = en_values[torch.nonzero(rad_bool)[:, 1]]
        rad_dist = dist_to_i[rad_bool]

        # compute wRDF
        rdf_diff_bins = torch.repeat_interleave(rad_dist, rdf_nbins).view(-1, rdf_nbins)
        rdf_gauss_bins = (-rdf_alpha * ((rdf_diff_bins - rdf_edges) ** 2)).exp()
        rdf = (rdf_gauss_bins.T * rad_p).sum(dim=1)

        # compute RWAAP
        wap_rad_numerator = rdf_gauss_bins.sum(dim=0)
        wap_rad = torch.zeros(rdf_nbins, dtype=torch.float)
        rad_mask = wap_rad_numerator != 0
        wap_rad[rad_mask] = rdf[rad_mask] / wap_rad_numerator[rad_mask]

        # compute wRACSF
        racsf_fcut = ((rad_dist * torch.pi / rad_cut).cos() + 1) * 0.5
        racsf_diff_tmp = torch.repeat_interleave(rad_dist, racsf_nbins)
        racsf_diff_bins = racsf_diff_tmp.view(-1, racsf_nbins) - racsf_edges
        racsf_gauss_bins = (-racsf_eta * (racsf_diff_bins**2)).exp().T
        racsf = (rad_p * racsf_fcut * racsf_gauss_bins).sum(dim=1)

        # initialize angular descriptors
        adf = torch.zeros(adf_nbins, dtype=torch.float)
        wap_ang = torch.zeros(adf_nbins, dtype=torch.float)
        aacsf = torch.zeros(aacsf_nbins, dtype=torch.float)

        # obtain distances within the cutoffs for angular descriptors
        ang_bool = (dist_to_i >= r_0) & (dist_to_i <= ang_cut)
        ang_p = en_values[torch.nonzero(ang_bool)[:, 1]]
        ang_dist = dist_to_i[ang_bool]

        if nad := len(ang_dist) > 1:
            # compute angles for all triplets (i,j,k) within the supercell
            ang_sphere_cart = cart_super[ang_bool]
            atoms_j, atoms_k = torch.combinations(torch.arange(nad, dtype=torch.long)).T
            property_jk = ang_p[atoms_j] * ang_p[atoms_k]
            ang_sphere_subtract = ang_sphere_cart[atoms_j] - ang_sphere_cart[atoms_k]
            d_ijk = torch.vstack(
                [
                    ang_dist[atoms_j],
                    ang_dist[atoms_k],
                    torch.linalg.norm(ang_sphere_subtract, dim=1),
                ]
            )
            d_ijk_sq = d_ijk**2
            ang_numberator = d_ijk_sq[0] + d_ijk_sq[1] - d_ijk_sq[2]
            ang_denominator = 2 * d_ijk[0] * d_ijk[1]
            cos_theta_ijk = ang_numberator / ang_denominator
            cos_theta_ijk[cos_theta_ijk < -1.0] = -1.0
            cos_theta_ijk[cos_theta_ijk > 1.0] = 1.0
            theta_ijk = cos_theta_ijk.arccos()

            # compute wADF
            adf_diff_bins = torch.tile(theta_ijk, (adf_nbins, 1)).T - adf_edges
            adf_gauss_bins = (-adf_beta * (adf_diff_bins**2)).exp()
            adf = (property_jk * adf_gauss_bins.T).sum(dim=1)

            # compute AWAAP
            wap_ang_numerator = adf_gauss_bins.sum(dim=0)
            ang_mask = wap_ang_numerator != 0
            wap_ang[ang_mask] = adf[ang_mask] / wap_ang_numerator[ang_mask]

            # compute wAACSF
            aacsf_fcut = (((d_ijk * torch.pi / ang_cut).cos() + 1) * 0.5).prod(dim=0)
            aacsf_pre = torch.tile(cos_theta_ijk, (aacsf_nbins, 1)).T
            aacsf_cos_bins = 1 + aacsf_lambda_bins * aacsf_pre
            aacsf_dist_bins = torch.tile(d_ijk_sq.sum(dim=0), (aacsf_nbins, 1)).T
            aacsf_gauss_bins = (-aacsf_eta_bins * aacsf_dist_bins).exp()
            aacsf_bins = (aacsf_cos_bins * aacsf_gauss_bins).T
            aacsf_bins *= property_jk * aacsf_fcut
            aacsf = aacsf_bins.sum(dim=1)

        features_list.append(torch.hstack([rdf, adf, racsf, aacsf, wap_rad, wap_ang]))

    return Data(
        x=torch.hstack([atom_features, torch.vstack(features_list)]),
        edge_index=torch.hstack(bond_pairs_list),
        edge_attr=torch.hstack(bond_dists_list).view(-1, 1),
    ).contiguous()
