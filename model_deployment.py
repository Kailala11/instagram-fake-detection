"""
Instagram Fake Account Detection - Model Deployment Script

This script demonstrates how to use the trained model for predictions
on new Instagram accounts.

Author: Data Science Portfolio Project
Date: November 2024
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

class FakeAccountDetector:
    """
    A class to detect fake Instagram accounts using trained ML model
    """
    
    def __init__(self, model_path='best_model.pkl', scaler_path='scaler.pkl'):
        """
        Initialize the detector with trained model and scaler
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model pickle file
        scaler_path : str
            Path to the fitted scaler pickle file
        """
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'profile_pic', 'nums_length_username', 'fullname_words', 
            'nums_length_fullname', 'name_username_match', 'description_length',
            'external_url', 'private', 'posts', 'followers', 'following',
            'follower_following_ratio', 'engagement_rate', 'activity_score',
            'profile_completeness', 'suspicious_username', 'follower_category'
        ]
        
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler from disk"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Model and scaler loaded successfully!")
            return True
        except FileNotFoundError:
            print("❌ Model or scaler file not found. Please train the model first.")
            return False
    
    def engineer_features(self, df):
        """
        Create engineered features from raw data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw account data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Follower-following ratio
        df_eng['follower_following_ratio'] = df_eng['followers'] / (df_eng['following'] + 1)
        
        # Engagement rate
        df_eng['engagement_rate'] = df_eng['posts'] / (df_eng['followers'] + 1)
        
        # Activity score
        df_eng['activity_score'] = (df_eng['posts'] > 0).astype(int) + \
                                    (df_eng['description_length'] > 0).astype(int)
        
        # Profile completeness
        df_eng['profile_completeness'] = (
            df_eng['profile_pic'] + 
            (df_eng['description_length'] > 0).astype(int) +
            df_eng['external_url']
        ) / 3
        
        # Suspicious username
        df_eng['suspicious_username'] = (df_eng['nums_length_username'] > 0.5).astype(int)
        
        # Follower category
        df_eng['follower_category'] = pd.cut(
            df_eng['followers'], 
            bins=[0, 100, 1000, 10000, 100000],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        return df_eng
    
    def predict_single(self, account_data):
        """
        Predict whether a single account is fake or real
        
        Parameters:
        -----------
        account_data : dict
            Dictionary containing account features
            
        Returns:
        --------
        dict
            Prediction results with probabilities
        """
        # Convert to DataFrame
        df = pd.DataFrame([account_data])
        
        # Engineer features
        df_eng = self.engineer_features(df)
        
        # Select required features
        X = df_eng[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': probabilities[prediction],
            'fake_probability': probabilities[1],
            'real_probability': probabilities[0],
            'risk_level': self._get_risk_level(probabilities[1])
        }
        
        return result
    
    def predict_batch(self, accounts_df):
        """
        Predict multiple accounts at once
        
        Parameters:
        -----------
        accounts_df : pandas.DataFrame
            DataFrame containing multiple account data
            
        Returns:
        --------
        pandas.DataFrame
            Original data with predictions
        """
        # Engineer features
        df_eng = self.engineer_features(accounts_df)
        
        # Select required features
        X = df_eng[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Add results to dataframe
        results = accounts_df.copy()
        results['prediction'] = ['FAKE' if p == 1 else 'REAL' for p in predictions]
        results['fake_probability'] = probabilities[:, 1]
        results['confidence'] = [probabilities[i][p] for i, p in enumerate(predictions)]
        results['risk_level'] = [self._get_risk_level(prob) for prob in probabilities[:, 1]]
        
        return results
    
    def _get_risk_level(self, fake_prob):
        """Determine risk level based on fake probability"""
        if fake_prob >= 0.8:
            return 'HIGH'
        elif fake_prob >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance_df
        else:
            print("Model does not support feature importance")
            return None


def example_usage():
    """Example of how to use the FakeAccountDetector class"""
    
    print("=" * 80)
    print("Instagram Fake Account Detection - Demo")
    print("=" * 80)
    
    # Initialize detector
    detector = FakeAccountDetector()
    
    # Note: In real usage, you would load the actual trained model
    # detector.load_model('best_model.pkl', 'scaler.pkl')
    
    # Example account 1 - Likely FAKE
    fake_account = {
        'profile_pic': 1,
        'nums_length_username': 0.6,
        'fullname_words': 1,
        'nums_length_fullname': 0.2,
        'name_username_match': 0,
        'description_length': 10,
        'external_url': 0,
        'private': 0,
        'posts': 5,
        'followers': 30,
        'following': 450
    }
    
    # Example account 2 - Likely REAL
    real_account = {
        'profile_pic': 1,
        'nums_length_username': 0.1,
        'fullname_words': 2,
        'nums_length_fullname': 0,
        'name_username_match': 0,
        'description_length': 120,
        'external_url': 1,
        'private': 0,
        'posts': 350,
        'followers': 25000,
        'following': 450
    }
    
    print("\n📊 Account Analysis:")
    print("-" * 80)
    
    # Analyze fake account
    print("\n🔍 Analyzing Account 1:")
    print(f"  Followers: {fake_account['followers']}")
    print(f"  Following: {fake_account['following']}")
    print(f"  Posts: {fake_account['posts']}")
    print(f"  Bio Length: {fake_account['description_length']}")
    
    # Calculate and show engineered features
    ratio_1 = fake_account['followers'] / (fake_account['following'] + 1)
    print(f"  Follower/Following Ratio: {ratio_1:.2f}")
    print("  ⚠️ LOW ratio (more following than followers) - SUSPICIOUS!")
    
    # Analyze real account
    print("\n🔍 Analyzing Account 2:")
    print(f"  Followers: {real_account['followers']}")
    print(f"  Following: {real_account['following']}")
    print(f"  Posts: {real_account['posts']}")
    print(f"  Bio Length: {real_account['description_length']}")
    
    ratio_2 = real_account['followers'] / (real_account['following'] + 1)
    print(f"  Follower/Following Ratio: {ratio_2:.2f}")
    print("  ✅ HIGH ratio (more followers than following) - LEGITIMATE!")
    
    print("\n" + "=" * 80)
    print("💡 Key Indicators:")
    print("=" * 80)
    print("FAKE Account Indicators:")
    print("  • Low follower count (< 100)")
    print("  • High following count (> 300)")
    print("  • Few posts (< 20)")
    print("  • Short or missing bio")
    print("  • Poor follower/following ratio")
    print("\nREAL Account Indicators:")
    print("  • Balanced follower/following ratio")
    print("  • Regular activity (50+ posts)")
    print("  • Complete profile with bio")
    print("  • Established follower base")
    
    print("\n" + "=" * 80)
    print("Note: Load trained model using detector.load_model() for actual predictions")
    print("=" * 80)


if __name__ == "__main__":
    # Run example
    example_usage()
    
    print("\n\n📝 Usage Instructions:")
    print("-" * 80)
    print("""
To use this script with trained model:

1. Train your model using the Jupyter notebook
2. Save the model and scaler as pickle files
3. Load and use:

    detector = FakeAccountDetector()
    detector.load_model('best_model.pkl', 'scaler.pkl')
    
    result = detector.predict_single(account_data)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    """)
