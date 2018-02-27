import pandas as pd
import numpy as np



def write_to_csv(test_x, predictions):
    submission = pd.DataFrame({
        "Id": test_x["Id"],
        "SalePrice": predictions
    })
    submission.to_csv('submission.csv', index=False)


def to_numerical(df, col):
    """Converts the categorical df[col] to numerical"""
    _, df[col] = np.unique(df[col], return_inverse=True)
    return df[col]


def max_scale(df, col):
    """Divides df[col] by its max value"""
    m = df[col].max()
    return df[col].multiply(1/m)


def convert_to(df, col, values, to):
    df[col] = df[col].replace(values, to)
    return df[col]


def drop_GrLivArea(X):
    X = X.drop(X[(X['GrLivArea'] > 4000)].index)

    return X



def engineer(X):
    X['2ndFlrSF'] = X['2ndFlrSF'].replace(np.NaN, 0)
    X['OvrFlrSF'] = X['1stFlrSF'] + X['2ndFlrSF']
    X['OvrFlrSF'] = max_scale(X, 'OvrFlrSF')

    X = X[['OverallQual', 'CentralAir', 'PoolQC', 'OvrFlrSF', 'SaleType', 'Neighborhood', 'ExterQual', 'SaleCondition', 'BsmtQual', 'TotalBsmtSF']]

    X['CentralAir'] = to_numerical(X, 'CentralAir')

    X['SaleType'] = convert_to(X, 'SaleType', ['New', 'Con'], 1.0)
    X['SaleType'] = convert_to(X, 'SaleType', ['CWD', 'ConLI'], 0.5)
    X['SaleType'] = convert_to(X, 'SaleType', ['WD', 'VWD', 'COD', 'ConLw', 'ConLD', 'Oth'], 0.0)

    X['SaleCondition'] = convert_to(X, 'SaleType', ['Partial'], 1.0)
    X['SaleCondition'] = convert_to(X, 'SaleType', ['Normal', 'Abnorml', 'Alloca', 'Family'], 0.5)
    X['SaleCondition'] = convert_to(X, 'SaleType', ['AdjLand', np.NaN], 0.00)

    X['PoolQC'] = convert_to(X, 'PoolQC', ['Ex', 'Gd', 'TA', 'Fa'], 1.0)
    X['PoolQC'] = X['PoolQC'].replace(np.NaN, 0.0)

    X['Neighborhood'] = convert_to(X, 'Neighborhood', ['NoRidge', 'NridgHt', 'StoneBr'], 1.0)
    X['Neighborhood'] = convert_to(X, 'Neighborhood', ['Veenker', 'Crawfor', 'CollgCr',
                                                                'Somerst', 'Timber', 'ClearCr',
                                                                'Blmngtn', 'Gilbert'], 0.5)
    X['Neighborhood'] = convert_to(X, 'Neighborhood', ['Mitchel', 'NWAmes', 'OldTown',
                                                                'BrkSide', 'Sawyer', 'SawyerW',
                                                                'IDOTRR', 'MeadowV', 'Edwards',
                                                                'NPkVill', 'BrDale', 'SWISU',
                                                                'Blueste', 'NAmes'], 0.0)

    X['ExterQual'] = convert_to(X, 'ExterQual', ['Ex'], 1.0)
    X['ExterQual'] = convert_to(X, 'ExterQual', ['Gd'], 0.5)
    X['ExterQual'] = convert_to(X, 'ExterQual', ['TA', 'Fa', 'Po'], 0.0)

    X['BsmtQual'] = X['BsmtQual'].replace(np.NaN, 'NA')
    X['BsmtQual'] = convert_to(X, 'BsmtQual', ['Ex'], 1.0)
    X['BsmtQual'] = convert_to(X, 'BsmtQual', ['Gd'], 0.5)
    X['BsmtQual'] = convert_to(X, 'BsmtQual', ['TA', 'Fa', 'Po', 'NA'], 0.0)
    
    X['TotalBsmtSF'] = X['TotalBsmtSF'].replace(np.NaN, 0)
    X['TotalBsmtSF'] = max_scale(X, 'TotalBsmtSF')

    # X.drop(['SaleType'], axis=1, inplace=True)

    return X