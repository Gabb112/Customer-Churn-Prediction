from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def resample_data(X_train, y_train, method='smote'):
    """Resamples the training data to address class imbalance."""
    if method == 'smote':
        resampler = SMOTE(random_state=42)
    elif method == 'undersample':
        resampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Invalid resampling method specified")
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled