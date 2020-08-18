from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# All sklearn Transforms must have the `transform` and `fit` methods
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self,strategy):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        for column in data.columns:  
            imputer = SimpleImputer(missing_values=np.nan,  strategy=self.strategy, verbose=0,copy=True)
            # Si la columna es de tipo objeto, se reemplazan los NAs por el valor más frecuente
            if data[column].dtype == np.dtype('O'):
                data[column] = data[column].fillna(data[column].value_counts().index[0])
            # Si la columna no es de tipo objeto, se reemplazan los NAs por la imputación con la estrategia indicada
            else: 
                data[column] = imputer.fit_transform(data[[column]])
        return data


# All sklearn Transforms must have the `transform` and `fit` methods
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        ""

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        for column in data.columns:  
            scaler = StandardScaler()
            # Si la columna no es de tipo objeto, se realiza es escalamiento
            if not data[column].dtype == np.dtype('O'):
                data[column] = scaler.fit_transform(data[[column]])
        return data
