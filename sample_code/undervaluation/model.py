from pyspark.ml.regression import LinearRegression

from pyspark.ml.feature import *
from builtins import round
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from datetime import *
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
import logging

import os
os.environ['ARROW_PRE_0_15_IPC_FORMAT']='1'
import sys
#sys.path.append('..')

from minval.minval.utilis.connect_spark_export import *
from minval.minval.utilis.get_param import *
from minval.minval.utilis.export_status import *

logger = logging.getLogger('MinVal.Model')

def create_features(df):
    """
        Cette fonction permet de créer un dictionaire de trois elements. 
        les cles sont le niveau de complexité de modelisation.
        les valeurs sont les noms des colonnes de features sui seront utilisés dans le niveau de complexité
        
        Input :
            df : dataframe d'entrée
            
        Output :
            un dictionnaire
    """
    
    list_Columns = df.columns
    features_full = ['l_qt_masse_nette'] + [cc for cc in list_Columns if cc.startswith('RegS_')]+ [cc for cc in list_Columns if cc.startswith('devise_')] + ['Pref_Tarif_1_digit', 'RepResponsabiliteDecl']
   
    features_inter = ['l_qt_masse_nette'] + [ cc for cc in list_Columns if cc.startswith('RegS_')]+ [ cc for cc in list_Columns if cc.startswith('devise_')]+['RepResponsabiliteDecl']
    
    features_base = ['l_qt_masse_nette'] + [ cc for cc in list_Columns if cc.startswith('RegS_')]

    dic_features ={"full": features_full, "inter":features_inter, "base":features_base}
    return(dic_features)


def typed_udf(return_type):
    """
    cette permet de definir le type de sortie d'une fonction
    """
    def _typed_udf_wrapper(func):
        return udf(func, return_type)
    return _typed_udf_wrapper

@typed_udf(ArrayType(StringType()))
def zip_udf(my_str) :
    """
    Cette fonction permet transformer  une chaine de characthere en une liste de liste.
        Example :  "A;B;C 1;2;3" ==> [[A, 1], [B 2], [C 3]]
        
        Input :
            my_str : chaine de charactère
        Output : 
            res : liste de liste
    """
    if my_str != None :
        strList = [x.split(";") for x in my_str.split(" ")]
        lenList = len(strList)
        subLenList = len(strList[0])
        if (lenList != 0) :
            res = [[strList[jj][ii] for jj in range (0, lenList)] for ii in range(0, subLenList)]
        return(res)


def extract_estimate(df, by_col, cols=['features', 'Coef', 'Std_Err', 'pvalues', 'Inf_95', 'Sup_95']):
    df=df.select([by_col]+cols).distinct()
    df=df.withColumn('concat_estimate', concat_ws(' ', col('features'), col('Coef'), col('Std_Err'), col('pvalues'), col('Inf_95'), col('Sup_95')))
    df=df.withColumn('zipped', zip_udf('concat_estimate'))\
        .withColumn('exploded', explode('zipped'))\
        .withColumn('clean_exploded', regexp_replace("exploded", "[\[\]]", ""))\
        .withColumn('explodedList', split('clean_exploded', ", "))
    new_columns = [by_col] + [(col('explodedList')[x]).alias(cols[x]) for x in range(0, len(cols))]
    df_out = df.select(new_columns)
    return df_out


def train_model(df, dict_features, by_col,  label='l_mt_valeur_stat'):
    """
    Cette fonction permet d'entrainer le modèle de regression lineaire
    
    Input 
        df : dataframe d'entree
        dict_features : dictionaire des features des niveaux de complexité de modele
        by_col : le nom de la colonne avec lequel les données vont être groupées
        label : le nom de la colonne de variable à modeliser. par defaut egale 'l_mt_valeur_stat'
    Output
        un dataframe avec les colonnes suivantes : id_article, by_col, r2, rmse, predict, beta, p_value, residus, 
    
    """
    
    ### definition de colonnes de metriques de modelisation
    df = df.withColumn('r2', lit(None).cast('double'))
    df = df.withColumn('rmse', lit(None).cast('double'))
    df = df.withColumn('predict', lit(None).cast('double'))
    df = df.withColumn('beta', lit(None).cast('double'))
    df = df.withColumn('p_value', lit(None).cast('double'))
    df = df.withColumn('residus', lit(None).cast('double'))
    df = df.withColumn('features', lit(None).cast(StringType()))\
            .withColumn('Coef', lit(None).cast(StringType()))\
            .withColumn('Std_Err', lit(None).cast(StringType()))\
            .withColumn('pvalues', lit(None).cast(StringType()))\
            .withColumn('Inf_95', lit(None).cast(StringType()))\
            .withColumn('Sup_95', lit(None).cast(StringType()))
    ### definition de schema de la detaframe de sortie
    
    schema = df.select('id_article', by_col,'r2', 'rmse', 'predict', 'beta', 'p_value', 'residus',
                       'features','Coef', 'Std_Err', 'pvalues', 'Inf_95', 'Sup_95').schema
    
    #### definition d'un enveloppe pour la fonction reg_lin qui prend un dataframe pandas. 
    #### cet enveloppe permet d'utliser la fonction que dataframe spark
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP) 
    def reg_lin(pdf):
        
        import os
        os.environ['ARROW_PRE_0_15_IPC_FORMAT']='1'
        Y = pdf[label]
        complexity_levels =list(dict_features.keys())
        for level in complexity_levels :
            features_model = dict_features[level]
            X = pdf[features_model]
            nbObs = X.shape[0]
            ncols = X.shape[1]
            if (nbObs>ncols):
                X2=sm.add_constant(X)
                model = sm.OLS(Y, X2)
                
                model=model.fit()
                p_values = list(model.pvalues)
                inf_10 = [x<0.1 for x in list(p_values)]
                if (all(inf_10)) | (level == complexity_levels[-1]) :
                    if (level == complexity_levels[-1]):                        
                        logger.info('W-Result conserved due to last complexity level')
                    p_value_masse_nette = model.pvalues['l_qt_masse_nette']
                    y_predict = model.predict()
                    residus = Y-y_predict
                    RMSE = rmse(Y, y_predict)
                    r2 = model.rsquared
                    beta_masse_nette = model.params['l_qt_masse_nette']
                    Values = model.summary2().tables[1]
                    
                    index=Values.index.values.tolist()
                    columns=['features']+Values.columns.values.tolist()
                    Vvalues=Values.values.tolist()
                    
                    
                    NewValues=[[index[ii]]+Vvalues[ii] for ii in range(0,len(index))]
                    
                    summary= pd.DataFrame(NewValues,  columns= columns)
                    features=";".join(summary['features'].astype("string").values)
                    coef=";".join(summary['Coef.'].astype("str").astype('string').values)
                    std=";".join(summary['Std.Err.'].astype("str").astype('string').values)
                    pv=";".join(summary['P>|t|'].astype("str").astype('string').values)
                    inf=";".join(summary['[0.025'].astype("str").astype('string').values)
                    sup=";".join(summary['0.975]'].astype("str").astype('string').values)
                else :
                    logger.info('The complexity '+level+ " don't conserved")
            else:
                r2 = None
                RMSE = None
                y_predict = None
                beta_masse_nette = None
                p_value_masse_nette = None
                residus = None
                features= None
                coef= None
                std= None
                pv= None
                inf= None
                sup= None
        df_out = pdf.assign(r2=r2, rmse=RMSE, predict=y_predict, beta=beta_masse_nette, p_value=p_value_masse_nette, residus=residus, features=features, Coef=coef, Std_Err=std, pvalues=pv, Inf_95=inf, Sup_95=sup)
        
        return df_out[['id_article', by_col, 'r2', 'rmse', 'predict', 'beta', 'p_value', 'residus',
                       'features','Coef', 'Std_Err', 'pvalues', 'Inf_95', 'Sup_95']]
    return df.groupby(by_col).apply(reg_lin)

def renamed_cols(df, cols, suffixe):
    """
    Cette fonction permet de renommer les colonnes 'cols' de la dataframe 'df' en rajoutant le suffixe 'suffixe'
    """
    df_out = df
    for cc in cols:
        df_out = df_out.withColumnRenamed(cc, cc+'_'+suffixe)
        
    return df_out
