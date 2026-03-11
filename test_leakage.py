import numpy as np
import os

def extract_group_id(file_path):
    """Parses the gXX group ID from a UCF50 filename."""
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    if len(parts) >= 3 and parts[-2].startswith('g'):
        return parts[-2]
    return "unknown_group"

def run_leakage_test():
    print("--- Running Data Leakage Test ---")
    
    # Load the saved splits
    if not os.path.exists('./splits.npy'):
        print("[Error] splits.npy not found. Did you run the training script?")
        return

    splits = np.load('./splits.npy', allow_pickle=True).item()
    
    # Extract just the file paths (ignoring the labels)
    train_paths = [sample[0] for sample in splits['train']]
    val_paths = [sample[0] for sample in splits['val']]
    test_paths = [sample[0] for sample in splits['test']]
    
    # Map paths to their unique Group IDs
    train_groups = set(extract_group_id(path) for path in train_paths)
    val_groups = set(extract_group_id(path) for path in val_paths)
    test_groups = set(extract_group_id(path) for path in test_paths)
    
    print(f"Total Unique Train Groups: {len(train_groups)}")
    print(f"Total Unique Val Groups:   {len(val_groups)}")
    print(f"Total Unique Test Groups:  {len(test_groups)}")
    
    # Calculate Intersections (Overlaps)
    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)
    
    # Assertions and Reporting
    leakage_found = False
    
    if len(train_val_overlap) > 0:
        print(f"\n[FAIL] Leakage detected between Train and Val! Shared groups: {train_val_overlap}")
        leakage_found = True
    else:
        print("\n[PASS] No overlap between Train and Validation sets.")
        
    if len(train_test_overlap) > 0:
        print(f"[FAIL] Leakage detected between Train and Test! Shared groups: {train_test_overlap}")
        leakage_found = True
    else:
        print("[PASS] No overlap between Train and Test sets.")
        
    if len(val_test_overlap) > 0:
        print(f"[FAIL] Leakage detected between Val and Test! Shared groups: {val_test_overlap}")
        leakage_found = True
    else:
        print("[PASS] No overlap between Validation and Test sets.")
        
    if not leakage_found:
        print("\n🎉 SUCCESS: The GroupShuffleSplit mathematically isolated all actors/environments. Your pipeline is 100% Leakage-Free!")

if __name__ == "__main__":
    run_leakage_test()