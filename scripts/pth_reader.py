"""Print Mellinger PID gains from a trained actor checkpoint as MATLAB code.

Replace --pth with the path to your best_actor.pth or last_actor.pth
(typically under log/<env>/<alg>/<run>/). Copy the printed Kp_lin ... Ki_rot
block into deploy/sim_to_sim/param_set.m, generate_mat_files.m, or
deploy/sim_to_real/FlyReferenceMellinger.m.

The paper actor stores the PID values directly (kp_xy, ki_z, ...). After each
actor step those parameters are projected onto the feasible-gain set; this
script does not re-apply bounds.

Example::

    python scripts/pth_reader.py --pth path/to/best_actor.pth
"""
import argparse
from pathlib import Path

import torch


def get_val(state_dict, key):
    """Read a scalar PID gain stored under its firmware-style name."""
    if key not in state_dict:
        raise KeyError(f"Cannot find '{key}' in state_dict.")
    return state_dict[key].item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Mellinger gains from an actor .pth as MATLAB assignments."
    )
    parser.add_argument(
        "--pth",
        type=Path,
        required=True,
        help="Path to best_actor.pth or last_actor.pth. Replace with your local checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pth_file = args.pth
    if not pth_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {pth_file}\n"
            "Set --pth to a local best_actor.pth or last_actor.pth."
        )

    state_dict = torch.load(pth_file, map_location="cpu")

    print(f"{'Parameter (key)':<30} | {'Shape':<15} | {'Preview'}")
    print("-" * 60)

    for key, value in state_dict.items():
        if value.numel() == 1:
            val_str = f"{value.item():.4f}"
        else:
            val_str = f"[Mean: {value.mean():.4f}]"
        print(f"{key:<30} | {str(list(value.shape)):<15} | {val_str}")

    print("\n% MATLAB gain block (paste into param_set.m, generate_mat_files.m, or FlyReferenceMellinger.m):")
    print("-" * 30)
    print(
        f"Kp_lin = [{get_val(state_dict, 'kp_xy'):.4f}; "
        f"{get_val(state_dict, 'kp_xy'):.4f}; "
        f"{get_val(state_dict, 'kp_z'):.4f}];"
    )
    print(
        f"Ki_lin = [{get_val(state_dict, 'ki_xy'):.4f}; "
        f"{get_val(state_dict, 'ki_xy'):.4f}; "
        f"{get_val(state_dict, 'ki_z'):.4f}];"
    )
    print(
        f"Kd_lin = [{get_val(state_dict, 'kd_xy'):.4f}; "
        f"{get_val(state_dict, 'kd_xy'):.4f}; "
        f"{get_val(state_dict, 'kd_z'):.4f}];"
    )
    print(
        f"Kr_rot = [{get_val(state_dict, 'kR_xy'):.2f}; "
        f"{get_val(state_dict, 'kR_xy'):.2f}; "
        f"{get_val(state_dict, 'kR_z'):.2f}];    % P term"
    )
    print(
        f"Kw_rot = [{get_val(state_dict, 'kw_xy'):.2f}; "
        f"{get_val(state_dict, 'kw_xy'):.2f}; "
        f"{get_val(state_dict, 'kw_z'):.2f}];    % D term"
    )
    print(
        f"Ki_rot = [{get_val(state_dict, 'ki_m_xy'):.4f}; "
        f"{get_val(state_dict, 'ki_m_xy'):.4f}; "
        f"{get_val(state_dict, 'ki_m_z'):.4f}];    % I term"
    )
    print("-" * 30)


if __name__ == "__main__":
    main()
