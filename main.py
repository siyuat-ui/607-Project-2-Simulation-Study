"""Main entry point for the engression simulation study.

This script runs the complete simulation study comparing engression-based
sample generation across different distributions.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation import SimulationExperiment, run_quick_simulation, run_full_simulation
from dgps import (
    NormalGenerator,
    ExponentialGenerator,
    UniformGenerator,
    LognormalGenerator,
    ChiSquareGenerator
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run engression simulation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test simulation (3 generators, small epochs)
  python main.py --mode quick --replications 3 --sizes 100 500
  
  # Run full simulation (all 5 generators, proper training)
  python main.py --mode full --replications 10 --sizes 100 500 1000
  
  # Run custom simulation
  python main.py --mode custom --generators normal exponential --replications 5 --sizes 200 400
  
  # Run with custom training parameters
  python main.py --mode quick --epochs 100 --batch-size 64 --lr 0.001
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'full', 'custom'],
        default='quick',
        help='Simulation mode: quick (fast test), full (all generators), or custom'
    )
    
    parser.add_argument(
        '--generators',
        type=str,
        nargs='+',
        choices=['normal', 'exponential', 'uniform', 'lognormal', 'chisquare', 'all'],
        default=['all'],
        help='Generators to use in custom mode'
    )
    
    parser.add_argument(
        '--replications',
        type=int,
        default=10,
        help='Number of replications per configuration (default: 10)'
    )
    
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[100, 500, 1000],
        help='Sample sizes to test (default: 100 500 1000)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 200 for full, 50 for quick)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training (default: 128)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: 1e-4 for full, 1e-3 for quick)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden layer dimension (default: 64)'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of hidden layers (default: 3)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout rate (default: 0.0)'
    )
    
    parser.add_argument(
        '--use-batchnorm',
        action='store_true',
        help='Use batch normalization'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    return parser.parse_args()


def get_generators(generator_names):
    """Get generator objects from names.
    
    Parameters
    ----------
    generator_names : list of str
        List of generator names
        
    Returns
    -------
    list of DataGenerator
        List of generator objects
    """
    generator_map = {
        'normal': NormalGenerator(loc=0, scale=1),
        'exponential': ExponentialGenerator(scale=1),
        'uniform': UniformGenerator(low=0, high=2),
        'lognormal': LognormalGenerator(mean=0, sigma=1),
        'chisquare': ChiSquareGenerator(df=5),
    }
    
    if 'all' in generator_names:
        return list(generator_map.values())
    
    return [generator_map[name] for name in generator_names if name in generator_map]


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("ENGRESSION SIMULATION STUDY")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Replications: {args.replications}")
    print(f"Sample sizes: {args.sizes}")
    print(f"Network architecture: {args.num_layers} layers × {args.hidden_dim} units")
    if args.dropout > 0:
        print(f"Dropout: {args.dropout}")
    if args.use_batchnorm:
        print(f"Batch normalization: enabled")
    print(f"Save results: {not args.no_save}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70 + "\n")
    
    # Run simulation based on mode
    if args.mode == 'quick':
        print("Running QUICK simulation (3 generators, fast training)...\n")
        
        # Set default parameters for quick mode if not specified
        epochs = args.epochs if args.epochs is not None else 50
        lr = args.lr if args.lr is not None else 1e-3
        
        results_df, summary_df = run_quick_simulation(
            n_replications=args.replications,
            sample_sizes=args.sizes,
            verbose=not args.quiet,
            save_results=not args.no_save
        )
    
    elif args.mode == 'full':
        print("Running FULL simulation (5 generators, proper training)...\n")
        
        # Set default parameters for full mode if not specified
        epochs = args.epochs if args.epochs is not None else 200
        lr = args.lr if args.lr is not None else 1e-4
        
        results_df, summary_df = run_full_simulation(
            n_replications=args.replications,
            sample_sizes=args.sizes,
            verbose=not args.quiet,
            save_results=not args.no_save
        )
    
    else:  # custom mode
        print(f"Running CUSTOM simulation...\n")
        
        # Get generators
        generators = get_generators(args.generators)
        print(f"Generators: {[g.name for g in generators]}\n")
        
        # Set training parameters
        epochs = args.epochs if args.epochs is not None else 200
        lr = args.lr if args.lr is not None else 1e-4
        
        training_params = {
            'num_epochs': epochs,
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'patience': args.patience,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'use_batchnorm': args.use_batchnorm,
        }
        
        # Run custom simulation
        sim = SimulationExperiment(
            generators=generators,
            sample_sizes=args.sizes,
            n_replications=args.replications,
            training_params=training_params,
            save_results=not args.no_save,
            results_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        results_df = sim.run_all_experiments()
        summary_df = sim.summarize_results(results_df)
    
    # Display final summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"Total experiments: {len(results_df)}")
    print(f"Unique generators: {results_df['generator'].nunique()}")
    print(f"Sample sizes tested: {sorted(results_df['sample_size'].unique())}")
    print()
    
    # Display summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 70)
    if not args.quiet:
        print(summary_df.to_string(index=False))
        print()
    
    # Key metrics summary
    print("\nKEY METRICS (averaged across all configurations):")
    print("-" * 70)
    print(f"Average MMD:           {results_df['mmd'].mean():.6f} ± {results_df['mmd'].std():.6f}")
    print(f"Average Mean Distance: {results_df['mean_distance'].mean():.6f} ± {results_df['mean_distance'].std():.6f}")
    print(f"Average Training Time: {results_df['training_time'].mean():.2f} ± {results_df['training_time'].std():.2f} seconds")
    print(f"Average Epochs:        {results_df['n_epochs'].mean():.1f} ± {results_df['n_epochs'].std():.1f}")
    print()
    
    if not args.no_save:
        print(f"Results saved to: {args.output_dir}/")
        print(f"  - Raw data: {args.output_dir}/raw/")
        print(f"  - Figures:  {args.output_dir}/figures/")
    
    print("=" * 70)
    
    return results_df, summary_df


if __name__ == "__main__":
    try:
        results_df, summary_df = main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)