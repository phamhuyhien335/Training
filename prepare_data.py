"""
Data preparation script for Plant Disease Detection.
This script handles data organization, sampling, and splitting.
"""

import os
import shutil
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import config
from utils import count_images_in_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directory_structure(output_dir: str, classes: List[str]):
    """
    Create directory structure for train/test splits.
    
    Args:
        output_dir: Base output directory
        classes: List of class names
    """
    for split in ['train', 'test']:
        for class_name in classes:
            dir_path = os.path.join(output_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
    
    logger.info(f"Created directory structure in {output_dir}")


def sample_and_split_data(
    source_dir: str,
    output_dir: str,
    classes: List[str],
    train_count: int,
    test_count: int,
    random_seed: int = None
):
    """
    Sample and split data into train and test sets.
    
    Args:
        source_dir: Source directory containing class subdirectories
        output_dir: Output directory for organized data
        classes: List of class names to process
        train_count: Number of samples per class for training
        test_count: Number of samples per class for testing
        random_seed: Random seed for reproducibility
    """
    if random_seed:
        random.seed(random_seed)
    
    stats = {
        'processed': {},
        'skipped': [],
        'errors': []
    }
    
    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.exists(source_class_dir):
            logger.warning(f"Class directory not found: {source_class_dir}")
            stats['skipped'].append(class_name)
            continue
        
        # Get all image files
        image_files = [
            f for f in os.listdir(source_class_dir)
            if os.path.isfile(os.path.join(source_class_dir, f))
            and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        total_available = len(image_files)
        total_needed = train_count + test_count
        
        if total_available < total_needed:
            logger.warning(
                f"{class_name}: Not enough images. "
                f"Available: {total_available}, Needed: {total_needed}"
            )
            # Use all available images and split proportionally
            train_count_adjusted = int(total_available * train_count / total_needed)
            test_count_adjusted = total_available - train_count_adjusted
        else:
            train_count_adjusted = train_count
            test_count_adjusted = test_count
        
        # Randomly sample images
        random.shuffle(image_files)
        train_files = image_files[:train_count_adjusted]
        test_files = image_files[train_count_adjusted:train_count_adjusted + test_count_adjusted]
        
        # Copy files to train directory
        train_dir = os.path.join(output_dir, 'train', class_name)
        for filename in train_files:
            src = os.path.join(source_class_dir, filename)
            dst = os.path.join(train_dir, filename)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.error(f"Error copying {src}: {e}")
                stats['errors'].append((src, str(e)))
        
        # Copy files to test directory
        test_dir = os.path.join(output_dir, 'test', class_name)
        for filename in test_files:
            src = os.path.join(source_class_dir, filename)
            dst = os.path.join(test_dir, filename)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.error(f"Error copying {src}: {e}")
                stats['errors'].append((src, str(e)))
        
        stats['processed'][class_name] = {
            'train': len(train_files),
            'test': len(test_files),
            'total_available': total_available
        }
        
        logger.info(
            f"Processed {class_name}: "
            f"Train={len(train_files)}, Test={len(test_files)}, "
            f"Available={total_available}"
        )
    
    return stats


def print_summary(stats: Dict):
    """
    Print summary of data preparation.
    
    Args:
        stats: Statistics dictionary
    """
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    
    print("\n📊 Processed Classes:")
    total_train = 0
    total_test = 0
    
    for class_name, counts in stats['processed'].items():
        print(f"\n  {class_name}:")
        print(f"    Train: {counts['train']}")
        print(f"    Test: {counts['test']}")
        print(f"    Available: {counts['total_available']}")
        total_train += counts['train']
        total_test += counts['test']
    
    print(f"\n📈 Totals:")
    print(f"  Total Train Images: {total_train}")
    print(f"  Total Test Images: {total_test}")
    print(f"  Total Images: {total_train + total_test}")
    
    if stats['skipped']:
        print(f"\n⚠️  Skipped Classes: {len(stats['skipped'])}")
        for class_name in stats['skipped']:
            print(f"    - {class_name}")
    
    if stats['errors']:
        print(f"\n❌ Errors: {len(stats['errors'])}")
        for src, error in stats['errors'][:5]:  # Show first 5 errors
            print(f"    - {src}: {error}")
        if len(stats['errors']) > 5:
            print(f"    ... and {len(stats['errors']) - 5} more errors")
    
    print("\n" + "="*60)


def verify_data_structure(data_dir: str):
    """
    Verify the data structure and print statistics.
    
    Args:
        data_dir: Base data directory
    """
    print("\n" + "="*60)
    print("DATA STRUCTURE VERIFICATION")
    print("="*60)
    
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            print(f"\n📁 {split.upper()} SET:")
            counts = count_images_in_directory(split_dir)
            total = sum(counts.values())
            print(f"  Total classes: {len(counts)}")
            print(f"  Total images: {total}")
            
            if counts:
                print(f"  Classes:")
                for class_name, count in sorted(counts.items()):
                    print(f"    - {class_name}: {count} images")
        else:
            print(f"\n⚠️  {split.upper()} directory not found: {split_dir}")
    
    print("\n" + "="*60)


def main():
    """
    Main data preparation function.
    """
    parser = argparse.ArgumentParser(description='Prepare Plant Disease Dataset')
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Source directory containing class subdirectories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.OUTPUT_DIR,
        help='Output directory for organized data'
    )
    parser.add_argument(
        '--train-count',
        type=int,
        default=config.TARGET_TRAIN_COUNT,
        help='Number of images per class for training'
    )
    parser.add_argument(
        '--test-count',
        type=int,
        default=config.TARGET_TEST_COUNT,
        help='Number of images per class for testing'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        help='Specific classes to process (if not specified, uses config)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=config.RANDOM_SEED,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data structure without processing'
    )
    
    args = parser.parse_args()
    
    # Verify only mode
    if args.verify_only:
        verify_data_structure(args.output_dir)
        return
    
    # Check source directory
    if not os.path.exists(args.source_dir):
        logger.error(f"Source directory not found: {args.source_dir}")
        return
    
    # Get classes to process
    if args.classes:
        classes = args.classes
    else:
        # Flatten classes from config
        classes = []
        for class_list in config.CLASSES_TO_PROCESS.values():
            classes.extend(class_list)
    
    logger.info(f"Processing {len(classes)} classes")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Train count per class: {args.train_count}")
    logger.info(f"Test count per class: {args.test_count}")
    
    # Create directory structure
    create_directory_structure(args.output_dir, classes)
    
    # Sample and split data
    logger.info("Starting data preparation...")
    stats = sample_and_split_data(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        classes=classes,
        train_count=args.train_count,
        test_count=args.test_count,
        random_seed=args.random_seed
    )
    
    # Print summary
    print_summary(stats)
    
    # Verify final structure
    verify_data_structure(args.output_dir)
    
    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()
