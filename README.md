# Análisis de Sentimiento de Valoración

<p align="center">

![imagen_readme](https://codenotch.com/blog/wp-content/uploads/2021/01/analisis-de-sentimientos.jpg)

</p>



Este proyecto final de máster tiene como objetivo realizar un análisis de sentimiento de valoración utilizando dos conjuntos de datos diferentes: uno de Twitter y otro de Amazon. El objetivo principal es analizar las opiniones y valoraciones de los usuarios en estas plataformas y determinar si son positivas, negativas o neutrales, además de la valoración en el caso de Amazon.

## Estructura del proyecto
El proyecto está organizado en diferentes carpetas y archivos, cada uno con un propósito específico. A continuación, se describe brevemente la estructura y contenido de cada uno:

Carpetas:

**Flask**: Contiene archivos relacionados con la implementación de una interfaz web utilizando el framework Flask. Se encarga de desplegar el modelo de análisis de sentimiento y mostrar los resultados de forma interactiva.

**bin/local**: Carpeta que contiene archivos relacionados con la configuración y ejecución del proyecto en un entorno local.

**data**: En esta carpeta se encuentran los conjuntos de datos utilizados en el proyecto, incluyendo el archivo de prueba añadido recientemente.

**models**: Contiene los modelos y scripts necesarios para llevar a cabo el análisis de sentimiento. También incluye archivos relacionados con el estilo y el análisis exploratorio de datos.

**notebooks**: Contiene archivos de Jupyter Notebook que muestran el proceso de desarrollo del proyecto y las correcciones realizadas recientemente.


## Software Utilizado

- Python 3.8
- Visual Studio Code
- Jupyter Notebook
- Flask

## ¿Cómo replicar el proyecto?


- 1. Instala el entorno virtual:

conda env create --name environment_name -f environment.yml

- 2. Introduce en la carpeta `data/raw` los siguientes archivos:

    - [Base de datos de Twitter](https://www.kaggle.com/datasets/kazanova/sentiment140/download?datasetVersionNumber=2)


    - [Base de datos Gift Cards de Amazon](https://forms.gle/UEkkJs69e7Z5A5Ps9)

## Autor

Victor Viloria Vázquez 
- Email: victor.viloria@cunef.edu
- Linkedin: https://www.linkedin.com/in/vicviloria/


## Link del proyecto:

https://github.com/ComputingVictor/TFM_2023
