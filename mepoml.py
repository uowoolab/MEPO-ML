from datetime import date
from pathlib import Path

import torch
from gemmi import UnitCell, cif
from torch_geometric.nn import GAT

from descriptors import featurize


def assign_single(src_path: Path, dst_path: Path, ap_dict: torch.Tensor, model: GAT):
    # keywords to look for in the CIF
    kla = "_cell_length_a"
    klb = "_cell_length_b"
    klc = "_cell_length_c"
    kaa = "_cell_angle_alpha"
    kab = "_cell_angle_beta"
    kag = "_cell_angle_gamma"
    kfx = "_atom_site_fract_x"
    kfy = "_atom_site_fract_y"
    kfz = "_atom_site_fract_z"
    kts = "_atom_site_type_symbol"

    # read the CIF with gemmi
    block = cif.read_file(str(src_path)).sole_block()
    vla = float(block.find_value(kla))
    vlb = float(block.find_value(klb))
    vlc = float(block.find_value(klc))
    vaa = float(block.find_value(kaa))
    vab = float(block.find_value(kab))
    vag = float(block.find_value(kag))
    cell = UnitCell(vla, vlb, vlc, vaa, vab, vag)
    vfx = [float(i) for i in block.find_loop(kfx)]
    vfy = [float(i) for i in block.find_loop(kfy)]
    vfz = [float(i) for i in block.find_loop(kfz)]
    frac_coords = torch.tensor([vfx, vfy, vfz], dtype=torch.float).T
    atom_symbols = [sym for sym in block.find_loop(kts)]

    # predict charges using the model
    # ====================
    graph_data = featurize(frac_coords, cell, atom_symbols, ap_dict)
    q_raw = model(**graph_data).flatten()
    q_final = q_raw - (q_raw.sum() / len(q_raw))

    # preambles for the new CIF
    cif_name = src_path.stem.removesuffix("_repeat")
    new_cif = f"""# Charges generated by MEPO-ML
data_{cif_name}
_audit_creation_date              {date.today().strftime("%Y-%m-%d")}
_audit_creation_method            MEPO-ML
_cell_length_a                    {vla:.6f}
_cell_length_b                    {vlb:.6f}
_cell_length_c                    {vlc:.6f}
_cell_angle_alpha                 {vaa:.6f}
_cell_angle_beta                  {vab:.6f}
_cell_angle_gamma                 {vag:.6f}
_cell_volume                      {cell.volume:.6f}
_symmetry_space_group_name_H-M    P1
_symmetry_Int_Tables_number       1
loop_
    _symmetry_equiv_pos_site_id
    _symmetry_equiv_pos_as_xyz
    1  x,y,z
loop_
    _atom_site_type_symbol
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_type_partial_charge
"""

    # create standardized atomic labels
    atom_labels = []
    label_counter = {element: 0 for element in set(atom_symbols)}
    for symbol in atom_symbols:
        label_counter[symbol] += 1
        atom_labels.append(f"{symbol}{label_counter[symbol]}")

    # compile all atomic information into a block of text for the new CIF
    symbol_width = len(max(atom_symbols, key=len))
    label_width = len(max(atom_labels, key=len))
    for symbol, label, fx, fy, fz, q in zip(
        atom_symbols, atom_labels, vfx, vfy, vfz, q_final
    ):
        new_cif += f"    {symbol:{symbol_width}}  {label:{label_width}}  "
        new_cif += f"{fx:.6f}  {fy:.6f}  {fz:.6f}  {q:.6f}\n"

    dst_path.joinpath(cif_name + "_mepoml.cif").write_text(new_cif)


def main(src_str, dst_str):
    # check existence of source
    assert src_str, "No source of CIF(s) given"
    src_path = Path(src_str)
    assert src_path.exists(), f"{src_str} DOES NOT EXIST."

    # given destination, mkdir if does not exist
    if dst_str is not None:
        dst_path = Path(dst_str)
        if not dst_path.is_dir():
            dst_path.mkdir(parents=True)
            print(f"Created new folder: {dst_path}")
    else:
        print(
            "No destination folder set, new CIF(s) will be stored in the source folder."
        )

    # load scaler and model
    ap_dict = torch.load("elemental_data.pt", weights_only=False)
    model_path = Path("mepoml_model.pt")
    model = GAT(
        in_channels=130,
        out_channels=1,
        v2=True,
        edge_dim=1,
        fill_value=0.0,
        hidden_channels=1024,
        num_layers=8,
        dropout=0.0,
        act="SiLU",
        heads=8,
        negative_slope=0.1,
        add_self_loops=True,
        residual=True,
        aggr="Median",
        norm="Layer",
        norm_kwargs={"mode": "node"},
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.inference_mode():
        # assign charges for a single CIF
        if src_path.is_file():
            # store new CIF in the same folder of the source if no dst given
            if dst_str is None:
                assign_single(src_path, src_path.parent, ap_dict, model)
            else:
                assign_single(src_path, dst_path, ap_dict, model)
        else:
            from tqdm import tqdm

            # assign charges for all CIFs in src folder
            # currently only running in serial due to some issues
            # when using multiprocessing with pytorch models
            src_glob = list(src_path.glob("*.cif"))
            assert len(src_glob), f"No CIFs found in {src_path}"
            for cif_path in tqdm(
                src_glob, ncols=0, ascii=True, desc="Assignment progress:"
            ):
                if dst_str is None:
                    assign_single(cif_path, src_path, ap_dict, model)
                else:
                    assign_single(cif_path, dst_path, ap_dict, model)


if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser(
        description="Predict REPEAT charges and output to a new CIF file."
    )
    ap.add_argument(
        "--src",
        type=str,
        metavar="SRC_DIR",
        help="Source directory for batch predictions",
    )
    ap.add_argument(
        "--dst",
        type=str,
        metavar="DST_DIR",
        help="Destination directory for the output CIF file",
    )
    args = ap.parse_args()
    main(args.src, args.dst)
