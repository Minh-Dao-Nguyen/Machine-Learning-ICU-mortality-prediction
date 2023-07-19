class Model():
    def __init__(self, args):
        from sklearn.ensemble import AdaBoostClassifier
        self.args = args
        self.ada = AdaBoostClassifier()
        

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        from sklearn.model_selection import GridSearchCV
        # Hypertuning
        param_grid = {'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
                      'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                      'algorithm': ['SAMME', 'SAMME.R']}

        # Train with grid search
        grid = GridSearchCV(self.ada, param_grid, cv=5)
        grid.fit(x_train, y_train)
        

    def predict_proba(self, x):

        preds = self.ada.predict_proba(x)[:, 1]

        return preds