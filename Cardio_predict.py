# 0.0 Import's
import pandas     as pd
import numpy      as np
import seaborn    as sns
import streamlit  as st

import io
import inflection
import pickle

from sklearn.ensemble        import RandomForestRegressor
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from tabulate                import tabulate
from matplotlib              import pyplot as plt
from IPython.display         import Image
from IPython.core.display    import HTML
from PIL                     import Image

import matplotlib.font_manager as fm

# 1.0. Importando Dataset's
df1 = pd.read_csv('cardio_test.csv', delimiter=';')

# 2.0 Funções de Limpeza/Pré Processamento

def cleaning (df1):
    # Verificar o comprimento dos valores na coluna 'ap_hi'
    comprimentos = df1['ap_hi'].astype(str).str.len()
    
    # Tratamento len = 1
    df1.loc[comprimentos == 1, 'ap_hi'] = df1.loc[comprimentos == 1, 'ap_hi'] * 10
    
    # Tratamento len = 2
    df1.loc[comprimentos == 2, 'ap_hi'] = (df1.loc[comprimentos == 2, 'ap_hi'] // 10).round() * 10
    
    # Tratamento len = 3
    df1.loc[comprimentos == 3, 'ap_hi'] = (df1.loc[comprimentos == 3, 'ap_hi'] // 10).round() * 10
    
    # Tratamento len = 4
    df1.loc[comprimentos == 4, 'ap_hi'] = (df1.loc[comprimentos == 4, 'ap_hi'] // 100).round() * 10
    
    # Tratamento len = 5
    df1.loc[comprimentos == 5, 'ap_hi'] = (df1.loc[comprimentos == 5, 'ap_hi'] // 1000).round() * 10
    
    # Verificar o comprimento dos valores na coluna 'ap_li'
    comprimentos = df1['ap_lo'].astype(str).str.len()
    
    # Tratamento len = 1
    df1.loc[comprimentos == 1, 'ap_lo'] = df1.loc[comprimentos == 1, 'ap_lo'] * 10
    
    # Tratamento len = 2
    df1.loc[comprimentos == 2, 'ap_lo'] = (df1.loc[comprimentos == 2, 'ap_lo'] // 10).round() * 10
    
    # Tratamento len = 3
    df1.loc[comprimentos == 3, 'ap_lo'] = (df1.loc[comprimentos == 3, 'ap_lo'] // 10).round() * 10
    
    # Tratamento len = 4
    df1.loc[comprimentos == 4, 'ap_lo'] = (df1.loc[comprimentos == 4, 'ap_lo'] // 100).round() * 10
    
    # Tratamento len = 5
    df1.loc[comprimentos == 5, 'ap_lo'] = (df1.loc[comprimentos == 5, 'ap_lo'] // 1000).round() * 10
    
    
    # Separando as variáveis em Númericas e Categóricas
    num_attributes = df1.drop(['id', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'], axis=1)
    cat_attributes = df1.drop(['age', 'height', 'weight', 'ap_hi', 'ap_lo'], axis=1)

    # Convertendo a coluna age em ano e removendo do dataframe desconsiderando da análise de outlier's 
    num_attributes['age'] = (num_attributes['age'] / 365).round().astype('int64')
     
    # Lista para armazenar os resultados
    results = []
    
    # Itera sobre as colunas do DataFrame
    for column in num_attributes.columns:
        # Calcula os quartis
        min =  num_attributes[column].min()
        max =  num_attributes[column].max()
        mean = num_attributes[column].mean().round()
        q1 = num_attributes[column].quantile(0.25)
        q3 = num_attributes[column].quantile(0.75)
    
        # Calcula o IQR
        iqr = q3 - q1
    
        # Calcula os limites inferior e superior
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
    
        # Filtra os valores que estão abaixo do limite inferior e acima do limite superior
        outliers_below = num_attributes[num_attributes[column] < lower_limit]
        outliers_above = num_attributes[num_attributes[column] > upper_limit]
    
        # Conta a quantidade de outliers abaixo e acima dos limites
        count_below = len(outliers_below)
        count_above = len(outliers_above)
    
        # Cria um dicionário com os resultados
        result = {
            'Feature': column,
            'Mínimo': min,
            'Máximo': max,
            'Média': mean,
            'Limite Inferior': lower_limit,
            'Limite Superior': upper_limit,
            'Outliers abaixo': count_below,
            'Outliers acima': count_above
        }
    
        # Adiciona o dicionário à lista de resultados
        results.append(result)
        
    # Definir os limites das faixas etárias
    age_bins = [30, 35, 40, 45, 50, 55, 60, 65, np.inf]
    
    # Definir os rótulos das faixas etárias
    age_labels = ['30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-70']
    
    # Adicionar uma nova coluna com as faixas etárias correspondentes
    num_attributes['age_group'] = pd.cut(num_attributes['age'], bins=age_bins, labels=age_labels, right=False)
    
    # Limites coluna height
    lower_limit = 145
    upper_limit = 200
    
    # função para substituir os valores fora do limite pela mediana da faixa etária correspondente
    def replace_outliers_height(row):
        if row['height'] < lower_limit or row['height'] > upper_limit:
            median_height = num_attributes.loc[num_attributes['age_group'] == row['age_group'], 'height'].median()
            return median_height
        return row['height']
    
    # Aplicando função
    num_attributes['height'] = num_attributes.apply(replace_outliers_height, axis=1)

    # Limites coluna weight
    lower_limit = 40.00
    upper_limit = 135.00
    
    # função para substituir os valores fora do limite pela mediana da faixa etária correspondente
    def replace_outliers_weight(row):
        if row['weight'] < lower_limit or row['weight'] > upper_limit:
            median_weight = num_attributes.loc[num_attributes['age_group'] == row['age_group'], 'weight'].median()
            return median_weight
        return row['weight']
    
    # Aplicando função
    num_attributes['weight'] = num_attributes.apply(replace_outliers_weight, axis=1)

    # Limites coluna ap_hi
    lower_limit = 70
    upper_limit = 300
    
    # função para substituir os valores fora do limite pela mediana da faixa etária correspondente
    def replace_outliers_ap_hi(row):
        if row['ap_hi'] < lower_limit or row['ap_hi'] > upper_limit:
            median_ap_hi = num_attributes.loc[num_attributes['age_group'] == row['age_group'], 'ap_hi'].median()
            return median_ap_hi
        return row['ap_hi']
    
    # Aplicando função
    num_attributes['ap_hi'] = num_attributes.apply(replace_outliers_ap_hi, axis=1)

    # Limites coluna ap_lo
    lower_limit = 50
    upper_limit = 200
    
    # função para substituir os valores fora do limite pela mediana da faixa etária correspondente
    def replace_outliers_ap_lo(row):
        if row['ap_lo'] < lower_limit or row['ap_lo'] > upper_limit:
            median_ap_lo = num_attributes.loc[num_attributes['age_group'] == row['age_group'], 'ap_lo'].median()
            return median_ap_lo
        return row['ap_lo']
    
    # Aplicando função
    num_attributes['ap_lo'] = num_attributes.apply(replace_outliers_ap_lo, axis=1)

    # Convertendo a coluna age em dias
    num_attributes['age'] = (num_attributes['age'] * 365).round().astype('int64')
    
    # Realizando merge entre os dataframes
    df1 = pd.concat([num_attributes, cat_attributes], axis=1)
    
    return df1

def feature_eng (df1):
    # Criação da nova coluna [IMC]
    df1['imc'] = (df1['weight'] / ((df1['height'] / 100) ** 2)).round(2)
    
    # Criação da nova coluna idade em anos [age_year]
    df1['age_year'] = (df1['age'] / 365).round().astype('int64')
    
    # Criação da nova coluna de classificação da obesidade [obesity_class]
    df1['obesity_class'] = pd.cut(df1['imc'], bins=[-float('inf'), 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
                                 labels=[-1, 0, 1, 2, 3, 4]).astype('int64')
    
    # Definir as faixas de pressão arterial por idade
    age_bins = [0, 18, 30, 40, 50, 60, np.inf]
    pressure_bins = [[85, 130], [85, 130], [90, 140], [100, 160], [110, 180], [90, 140]]
    
    # Função para classificar a pressão arterial
    def classify_pressure(row):
        age = row['age_year']
        sys_pressure = row['ap_hi']
        dia_pressure = row['ap_lo']
        
        # Encontrar a faixa etária correspondente
        age_group = pd.cut([age], bins=age_bins, labels=False, right=False)[0]
        
        # Encontrar a classificação da pressão arterial
        pressure_class = 0
        sys_min, sys_max = pressure_bins[age_group]
        dia_min, dia_max = pressure_bins[age_group]
        if sys_min <= sys_pressure <= sys_max and dia_min <= dia_pressure <= dia_max:
            pressure_class = 1
        
        return pressure_class
    
    # Criar a nova coluna de classificação da pressão arterial
    df1['pressure_class'] = df1.apply(classify_pressure, axis=1)
    
    return df1

def feature_pre_processing(df1):
    # Aplicar transformação logarítmica
    df1['imc'] = np.log(df1['imc'])
    df1['weight'] = np.log(df1['weight'])
    df1['height'] = np.log(df1['height'])

    cols_not_selected = df1.drop(columns=['height', 'weight', 'ap_hi', 'ap_lo', 'age_year', 'imc', 'cholesterol'])
    cols_selected = df1.drop(columns=cols_not_selected.columns)
    
    return cols_selected, cols_not_selected

# 3.0. Carregando modelo Treinado
model = pickle.load( open( 'model_cardio.pkl', 'rb'))

# 4.0. Aplicando a Predição

# Pré-processamento
df_processed = cleaning(df1)
df_processed = feature_eng(df_processed)
cols_selected, cols_not_selected = feature_pre_processing(df_processed)

# Realizar as predições
X_test = cols_selected.copy()
predictions = model.predict(X_test)

# Adicionar as predições ao dataframe
cols_selected['predict'] = predictions

# Reverter transformação logarítmica
cols_selected['imc'] = np.expm1(cols_selected['imc']).round(2)
cols_selected['weight'] = np.expm1(cols_selected['weight']).round(2)
cols_selected['height'] = (np.expm1(cols_selected['height']) / 100).round(2)

# Realizar o merge com as colunas não selecionadas
df_merged = pd.concat([cols_not_selected, cols_selected], axis=1)

# Carregar o arquivo "cardio_data" em um DataFrame
df_cardio_data = pd.read_csv('cardio_data.csv', delimiter=';')

# Selecionar colunas relevantes do "df_cardio_data"
df_cardio_data = df_cardio_data[['id', 'email', 'type_plan', 'city']]

# Realizar a mesclagem (merge) entre os DataFrames
df_final = pd.merge(df_merged, df_cardio_data, on='id')

# Reordenar as colunas do DataFrame df_final
column_order = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_year',
                'age_group', 'imc', 'obesity_class', 'pressure_class', 'email', 'type_plan', 'city', 'predict']

df_final = df_final.reindex(columns=column_order)

# Salvar o DataFrame df_final em um arquivo Excel
df_final.to_excel('cardio_predictions.xlsx', index=False)

#===================================================== ### VISUALIZAÇÃO DOS RESULTADOS ### ================================================================#
#___________________________________________#_________________________________________________#

# Criação Nome da Página
st.set_page_config(page_title = 'Predição Doenças Cardio')

# Importanto Logo
image = Image.open('cardio_ft.png')

# Definir a barra lateral (sidebar) no Streamlit
st.sidebar.image( image, width = 180)
st.sidebar.markdown('# Predição Doenças Cardiovasculares')
st.sidebar.markdown("""---""")

# Função para filtrar o DataFrame com base no tipo de visão selecionado
def filter_data(df_final, visao):
    if 'Cardio' in visao and 'Não Cardio' in visao:
        return df_final  # Retorna o DataFrame completo se ambas as visões estiverem selecionadas
    elif 'Cardio' in visao:
        return df_final[df_final['predict'] == 1]  # Retorna apenas os pacientes cardio
    elif 'Não Cardio' in visao:
        return df_final[df_final['predict'] == 0]  # Retorna apenas os pacientes não cardio
    else:
        return df_final

# Obter o tipo de visão selecionado
visao = st.sidebar.multiselect('Escolha o Tipo de Visão', ['Cardio', 'Não Cardio'])

# Filtrar o DataFrame com base no tipo de visão selecionado
df_filtered = filter_data(df_final, visao)    
    
# Converter o DataFrame para um arquivo Excel em memória
excel_buffer = io.BytesIO()
df_final.to_excel(excel_buffer, index=False)
excel_data = excel_buffer.getvalue()
st.sidebar.markdown("""---""")

# Definir Opção de Download
st.sidebar.title('Opções')
st.sidebar.write('Clique no botão abaixo para fazer o download do arquivo Excel.')
st.sidebar.download_button(label='Download Excel', data=excel_data, file_name='df_final.xlsx')

st.sidebar.markdown("""---""")
st.sidebar.markdown('## Powered By Luiz Zamian')

# Definição Estrutura principal Dash
st.write(" ## Predição de Doenças Cardiovasculares")

st.markdown("""---""")

# Total de Pacientes
total_pacientes = df_filtered.shape[0]

# Predição Cardio
predicao_cardio = df_filtered['predict'].sum()

# Pacientes Não Cardio
pacientes_nao_cardio = total_pacientes - predicao_cardio

# Precisão do Algoritmo
precisao_algoritmo = 75.00

# Média Pressão Sistólica
mean_sist = df_filtered['ap_hi'].mean().round()

# Média Pressão Sistólica
mean_diast = df_filtered['ap_lo'].mean().round()

# Criação dos rótulos
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total de Pacientes", value=total_pacientes)
with col2:
    st.metric(label="Predição Cardio", value=predicao_cardio)
with col3:
    st.metric(label="Pacientes Não Cardio", value=pacientes_nao_cardio)
with col4:
    st.metric(label="Precisão do Algoritmo", value="{:.2f}%".format(precisao_algoritmo))


# Criação dos rótulos
col5, col6 = st.columns(2)

with col5:
    st.metric(label="Média Pressão Sistólica", value=mean_sist)
with col6:
    st.metric(label="Média Pressão Diastólica", value=mean_diast)


# Dicionário de mapeamento para as legendas
legend_mapping = {0: "Não Cardio", 1: "Cardio"}

# Configurar as cores dos gráficos
colors = []
if 'Cardio' in visao and 'Não Cardio' in visao:
    colors = ['blue', 'red']  # Cardio = vermelho, Não Cardio = azul
elif 'Cardio' in visao:
    colors = ['red']  # Cardio = vermelho
elif 'Não Cardio' in visao:
    colors = ['blue']  # Não Cardio = azul
else:
    colors = ['blue', 'red']  # Padrão: Cardio = vermelho, Não Cardio = azul

# Criação de quadrantes
col2, col3 = st.columns(2)

# Tamanho dos gráficos
fig_width = 10
fig_height = 10

# Tamanho da fonte dos rótulos
label_fontsize = 18

# Tamanho da fonte dos valores percentuais
autopct_fontsize = 18

# Gráfico de Pizza
with col2:
    st.header("Pacientes Cardio x Não Cardio ")
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    # Substituir os valores no DataFrame
    df_final_mapped = df_filtered.replace({"predict": legend_mapping})

    colors = colors
    pie = df_final_mapped['predict'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1,
                                                        fontsize=autopct_fontsize, pctdistance=0.85,
                                                        colors=colors, textprops={'color': 'white'})
    ax1.set_aspect('equal')
    ax1.set_ylabel('')
    ax1.legend(fontsize=label_fontsize)

    # Definir a cor de fundo do gráfico de pizza como branco
    pie.set_facecolor('white')

    st.pyplot(fig1, clear_figure=True)
    
# Gráfico de Colunas Empilhadas
with col3:
    st.header("Predição s/ Faixa Etária")
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.groupby(['age_group', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax2, color=colors)
    ax2.set_xlabel('Age Group', fontsize=label_fontsize)
    ax2.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax2.legend(fontsize=label_fontsize)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=label_fontsize)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig2, clear_figure=True)
    
# Criação dos quadrantes
col4, col5 = st.columns(2)

# Gráfico de Colunas Empilhadas
with col4:
    st.header("Predição s/ Obesidade")
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.groupby(['obesity_class', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax3, color=colors)
    ax3.set_xlabel('Obesity Class', fontsize=label_fontsize)
    ax3.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax3.legend(fontsize=label_fontsize)
    ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=label_fontsize)
    ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig3, clear_figure=True)

# Gráfico de Colunas Empilhadas

# Dicionário de mapeamento para as legendas
gender_mapping = {1: "Masculino", 2: "Feminino"}

with col5:
    st.header("Predição s/ Gênero")
    fig4, ax4 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.replace({"gender": gender_mapping}).groupby(['gender', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax4, color=colors)
    ax4.set_xlabel('Gênero', fontsize=label_fontsize)
    ax4.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax4.legend(fontsize=label_fontsize)
    ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=label_fontsize)
    ax4.set_yticklabels(ax4.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig4, clear_figure=True)

# Criação dos quadrantes
col6, col7 = st.columns(2)
    
# Gráfico de Colunas Empilhadas
with col6:
    st.header("Predição s/ Tipo de Plano")
    fig5, ax5 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.groupby(['type_plan', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax5, color=colors)
    ax5.set_xlabel('Tipo de Plano', fontsize=label_fontsize)
    ax5.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax5.legend(fontsize=label_fontsize)
    ax5.set_xticklabels(ax5.get_xticklabels(), fontsize=label_fontsize)
    ax5.set_yticklabels(ax5.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig5, clear_figure=True)
    
# Gráfico de Colunas Empilhadas
with col7:
    st.header("Predição s/ Tipos de Cidade")
    fig6, ax6 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.groupby(['city', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax6, color=colors)
    ax6.set_xlabel('Cidades', fontsize=label_fontsize)
    ax6.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax6.legend(fontsize=label_fontsize)
    ax6.set_xticklabels(ax6.get_xticklabels(), fontsize=label_fontsize)
    ax6.set_yticklabels(ax6.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig6, clear_figure=True)
    
# Criação dos quadrantes
col8, col9, col10 = st.columns(3)

# Gráfico de Colunas Empilhadas
with col8:
    st.header("Predição s/ Glicose")
    fig7, ax7 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.groupby(['gluc', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax7, color=colors)
    ax7.set_xlabel('Glicose', fontsize=label_fontsize)
    ax7.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax7.legend(fontsize=label_fontsize)
    ax7.set_xticklabels(ax7.get_xticklabels(), fontsize=label_fontsize)
    ax7.set_yticklabels(ax7.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig7, clear_figure=True)
    
# Gráfico de Colunas Empilhadas

# Dicionário de mapeamento para as legendas
smoke_mapping = {0: "Não Fumante", 1: "Fumante"}

with col9:
    st.header("Predição s/ Fumantes")
    fig8, ax8 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.replace({"smoke": smoke_mapping}).groupby(['smoke', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax8, color=colors)
    ax8.set_xlabel('Fumantes', fontsize=label_fontsize)
    ax8.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax8.legend(fontsize=label_fontsize)
    ax8.set_xticklabels(ax8.get_xticklabels(), fontsize=label_fontsize)
    ax8.set_yticklabels(ax8.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig8, clear_figure=True)

# Gráfico de Colunas Empilhadas

# Dicionário de mapeamento para as legendas
alco_mapping = {0: "Não Álcool", 1: "Álcool"}

with col10:
    st.header("Predição s/ Álcool")
    fig9, ax9 = plt.subplots(figsize=(fig_width, fig_height))
    df_grouped = df_filtered.replace({"alco": alco_mapping}).groupby(['alco', 'predict']).size().unstack()
    df_grouped.rename(columns=legend_mapping, inplace=True)  # Renomear as colunas
    df_grouped.plot(kind='bar', stacked=True, ax=ax9, color=colors)
    ax9.set_xlabel('Álcool', fontsize=label_fontsize)
    ax9.set_ylabel('Qtd Pacientes', fontsize=label_fontsize)
    ax9.legend(fontsize=label_fontsize)
    ax9.set_xticklabels(ax9.get_xticklabels(), fontsize=label_fontsize)
    ax9.set_yticklabels(ax9.get_yticklabels(), fontsize=label_fontsize)
    
    st.pyplot(fig9, clear_figure=True)