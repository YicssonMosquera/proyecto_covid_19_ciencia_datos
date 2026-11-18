# Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial
**Universidad de La Salle — Facultad de Ingeniería / Maestría en Inteligencia Artificial**

## Descripción
Sistema analítico integral para el monitoreo epidemiológico global (189 países), auditoría de limpieza de datos, predicción continua de mortalidad diaria (`Random Forest Regressor`, $R^2 = 0.927$), clasificación temprana de riesgo sanitario (`Random Forest Classifier`, $Recall = 93.8\%$) y segmentación territorial no supervisada (`K-Means + PCA`).

## Estructura del Repositorio
- `/data`: Conjunto de datos original (`72.090` registros) y tablas procesadas limpias (`51.030` registros nacionales).
- `/notebooks`: Cuaderno de Google Colab con las Fases 1, 2 y 3.
- `/src`: Script modular de entrenamiento (`train_pipeline.py`).
- `/dashboard`: Aplicación web interactiva en Streamlit (`app.py`) y modelos serializados (`.pkl`).

## Despliegue en Vivo
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
