# Screenshots pendientes — captura manual

No hay playwright ni selenium en el entorno. Capturar manualmente
con el dashboard corriendo (`streamlit run src/app/main.py`).

## Lista de screenshots requeridos

| # | Archivo | Que capturar |
|---|---------|-------------|
| 1 | `tab_backtest_full.png` | Tab Backtest completo con tabla de 7 estrategias visible |
| 2 | `tab_backtest_chart.png` | Solo el grafico de costo por estrategia (zoom in) |
| 3 | `tab_alertas_fold1.png` | Tab Alertas con fold 1, mostrando alertas, decisiones y capacidad por CEDI |
| 4 | `tab_chat_demo.png` | Tab Chat con conversacion de ejemplo (Gemini real si hay key, mock si no) |
| 5 | `dashboard_full_sidebar.png` | Vista general con sidebar y metricas headline visibles |

## Instrucciones

1. `cd c:\Ecosfera\Practica-herdez`
2. `uv run streamlit run src/app/main.py`
3. Abrir http://localhost:8501
4. Capturar cada screenshot (Win+Shift+S o herramienta preferida)
5. Guardar en este directorio con los nombres exactos de la tabla
