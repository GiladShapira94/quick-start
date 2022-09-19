from sklearn import ensemble
from sklearn.model_selection import train_test_split
from mlrun.frameworks.sklearn import apply_mlrun
import mlrun

def train_iris(dataset: mlrun.DataItem, label_column: str):
    
    # Initialize our dataframes
    df = dataset.as_df()
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Train/Test split Iris data-set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier()
    
    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model=model, model_name='my_model', x_test=X_test, y_test=y_test)
    
    # Train our model
    model.fit(X_train, y_train)