import os
import pandas as pd

'''
Utility functions for competition submission
'''

def save_submission(test, preds, output_dir):
    '''
    Saves predictions in Kaggle submission format
    
    Args:
        test: Test DataFrame
        preds: Model predictions
        output_dir: Directory to save submission file
    '''
    submission = pd.DataFrame({
        'id': test['id'],
        'sii': preds
    })
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")