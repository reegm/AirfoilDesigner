import argparse
import os
import sys
import logging

# Central configuration
from core import config
from core.core_logic import CoreProcessor
from utils.dxf_exporter import export_curves_to_dxf # Import dxf_exporter to save the doc

def main():
    """
    Main function for the command-line airfoil processing application.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(
        description="""
        A command-line tool for processing airfoil data files.
        This tool loads an airfoil .dat file, optimizes it using a 4-segment Bezier curve model,
        optionally refines the model by adding control points, combines the segments into two
        single Bezier curves, and exports the result as a DXF file.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "dat_file",
        type=str,
        help="Path to the input airfoil .dat file."
    )
    parser.add_argument(
        "-t",
        "--te_thickness",
        type=float,
        default=config.DEFAULT_TE_THICKNESS_PERCENT,
        help=(
            "Desired trailing edge thickness as a percentage of the chord length (e.g., 0.5 for 0.5%%). "
            f"Defaults to {config.DEFAULT_TE_THICKNESS_PERCENT}."
        ),
    )
    parser.add_argument(
        "-c",
        "--chord_length",
        type=float,
        default=config.DEFAULT_CHORD_LENGTH_MM,
        help=(
            "Desired chord length in millimeters for the final DXF output. "
            f"Defaults to {config.DEFAULT_CHORD_LENGTH_MM} mm."
        ),
    )
    parser.add_argument(
        "-f", "--output_file",
        type=str,
        default=None,
        help="Filename for the output DXF file. Defaults to the input filename with a .dxf extension."
    )
    parser.add_argument(
        "--refinement_steps",
        type=int,
        default=5,
        help="Number of refinement steps to run. Set to 0 to disable.",
    )
    parser.add_argument(
        '-m', '--merge-segments',
        action='store_true',
        help="Merge the four Bezier segments into two (upper and lower) before exporting."
    )
    parser.add_argument(
        "-sm", "--smoothness_weight",
        type=float,
        default=0.005,
        help="Weight for smoothness penalty of the Bezier curves. Defaults to 0.005."
    )
    parser.add_argument(
        "-sp", "--spacing_weight",
        type=float,
        default=0.01,
        help="Weight for spacing the control points of the Bezier curves. Defaults to 0.01."
    )
    parser.add_argument(
        "-reg", "--regularization_weight",
        type=float,
        default=0.01,
        help="Regularization weight for single Bezier model. Defaults to 0.01."
    )

    args = parser.parse_args()

    # --- Validate input file ---
    if not os.path.exists(args.dat_file):
        logging.error(f"Error: Input file not found at '{args.dat_file}'")
        sys.exit(1)

    # --- Determine output filename ---
    output_filename = args.output_file

    # --- Initialize and run the core processor ---
    processor = CoreProcessor()

    # Run the full process
    success = processor.run_full_process(
        dat_file=args.dat_file,
        output_filename=output_filename,
        chord_length_mm=args.chord_length,
        refinement_steps=args.refinement_steps,
        smoothness_weight=args.smoothness_weight,
        spacing_weight=args.spacing_weight,
        te_thickness_percent=args.te_thickness,
        regularization_weight=args.regularization_weight,
        export_single_bezier_model=args.merge_segments
    )
    if not success:
        logging.error("Processing failed. See log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
