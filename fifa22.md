# Exploración de Datos FIFA22

# 1. Introducción

Este análisis tiene como objetivo explorar el conjunto de datos de FIFA22, 
centrándose en variables clave como el potencial, salario, reputación, valor de mercado y estatura de los jugadores.

```python
# 2. Carga y exploración inicial de datos

# 2.1 Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# 2.2 Carga del dataset
df = pd.read_csv('ruta/a/tu/archivo.csv')
```

```python
# 2.3 Exploración inicial
df.head()
```

```python
df.info()
```

```python
df.describe()
```

```python
# 2.4 Identificación de valores nulos
df.isnull().sum()
```

# 3. Análisis exploratorio (EDA)

## 3.1 Distribución de características físicas y económicas

```python
# Estatura de los jugadores
sns.histplot(df['Height'], kde=True)
plt.title('Distribución de Estatura')
plt.show()
```

```python
# Peso de los jugadores
sns.histplot(df['Weight'], kde=True)
plt.title('Distribución de Peso')
plt.show()
```

```python
# Salario de los jugadores
sns.histplot(df['Wage'], kde=True)
plt.title('Distribución de Salario')
plt.show()
```

```python
# Edad de los jugadores
sns.histplot(df['Age'], kde=True)
plt.title('Distribución de Edad')
plt.show()
```

## 3.2 Dominancia del pie

```python
sns.countplot(x='Preferred Foot', data=df)
plt.title('Pie Dominante de los Jugadores')
plt.show()
```

## 3.3 Representación geográfica

```python
top_countries = df['Nationality'].value_counts().head(10)
top_countries.plot(kind='bar')
plt.title('Países con Mayor Representación')
plt.show()
```

## 3.4 Posiciones de los jugadores

```python
sns.countplot(y='Position', data=df, order=df['Position'].value_counts().index)
plt.title('Posiciones más Comunes')
plt.show()
```

# 4. Relaciones entre variables clave

## 4.1 Potencial vs Salario

```python
sns.scatterplot(x='Potential', y='Wage', data=df)
plt.title('Relación entre Potencial y Salario')
plt.show()
```

## 4.2 Reputación vs Valor de mercado

```python
sns.scatterplot(x='Reputation', y='Value', data=df)
plt.title('Relación entre Reputación y Valor de Mercado')
plt.show()
```

## 4.3 Reputación vs Salario

```python
sns.boxplot(x='Reputation', y='Wage', data=df)
plt.title('Relación entre Reputación y Salario')
plt.show()
```

## 4.4 Estatura vs Potencial

```python
sns.scatterplot(x='Height', y='Potential', data=df)
plt.title('Relación entre Estatura y Potencial')
plt.show()
```

# 5. Jugadores destacados

## 5.1 Jugadores mejor remunerados

```python
df[['Name', 'Wage']].sort_values(by='Wage', ascending=False).head(10)
```

## 5.2 Jugadores con mayor potencial

```python
df[['Name', 'Potential']].sort_values(by='Potential', ascending=False).head(10)
```

## 5.3 Jugadores con mejor valoración general

```python
df[['Name', 'Overall']].sort_values(by='Overall', ascending=False).head(10)
```

# 6. Conclusiones

- Se identificaron patrones interesantes en las relaciones entre potencial, salario y reputación.
- La mayoría de los jugadores tienden a ser diestros.
- Determinados países dominan la representación de jugadores.
- Hay correlaciones positivas moderadas entre reputación y valor de mercado.
- Se pueden proponer futuros estudios como predicción del valor de mercado a partir de atributos físicos y técnicos.
