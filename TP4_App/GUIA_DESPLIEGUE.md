# 🚀 Guía de despliegue — App de Riesgo de ACV (Streamlit Community Cloud)

Esta guía te lleva paso a paso para publicar la app online y gratis. Necesitás una cuenta
de **GitHub** (gratuita) y una de **Streamlit Community Cloud** (se crea con tu GitHub).

---

## Paso 0 — Probar localmente (recomendado antes de subir)

> **¿Dónde escribo estos comandos?** En una **terminal / línea de comandos** de tu
> computadora (no en Python ni en Jupyter):
> - **Windows:** abrí el menú Inicio, escribí `cmd` y abrí "Símbolo del sistema"
>   (o "PowerShell"). También podés abrir la carpeta `TP4_App` en el Explorador, hacer clic
>   en la barra de direcciones, escribir `cmd` y Enter: la terminal se abre ya ubicada ahí.
> - **Mac:** abrí la app **Terminal** (Spotlight → "Terminal").
> - **Requisito previo:** tener Python instalado (probá `python --version`).
>
> El comando `cd TP4_App` te mueve hasta la carpeta del proyecto. Si la carpeta está, por
> ejemplo, en Descargas, primero corré `cd "C:\Users\monte\Downloads\JuanCruz\Inteligencia Artifical\TP4_App"`
> (con la ruta completa entre comillas) y después los otros dos comandos.

```bash
cd TP4_App
pip install -r requirements.txt
streamlit run app.py
```

Si abre en `http://localhost:8501` y predice bien, está lista para subir.

---

## Paso 1 — Crear el repositorio en GitHub

1. Entrá a <https://github.com> e iniciá sesión (o creá una cuenta).
2. Botón **"New repository"** (arriba a la derecha, ícono `+`).
3. Nombre sugerido: `tp4-stroke-app`. Marcalo como **Public**. No agregues README (ya lo tenés).
4. Click en **"Create repository"**.

## Paso 2 — Subir los archivos

**Opción A — desde la web (sin instalar nada):**
1. En el repo recién creado, click **"uploading an existing file"**.
2. Arrastrá **todo el contenido de la carpeta `TP4_App`**:
   `app.py`, `train_model.py`, `stroke_model.joblib`,
   `healthcare-dataset-stroke-data.csv`, `requirements.txt`, `README.md`.
3. Escribí un mensaje (ej: "primera versión") y **"Commit changes"**.

**Opción B — con Git desde la terminal:**
```bash
cd TP4_App
git init
git add .
git commit -m "App de predicción de ACV - TP4 Grupo 7"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/tp4-stroke-app.git
git push -u origin main
```

> ⚠️ Verificá que `stroke_model.joblib` **sí** se haya subido (es el modelo). Si usás un
> `.gitignore`, no excluyas ese archivo.

## Paso 3 — Desplegar en Streamlit Community Cloud

1. Entrá a <https://share.streamlit.io> y **"Sign in with GitHub"** (autorizá el acceso).
2. Click en **"Create app"** → **"Deploy a public app from GitHub"**.
3. Completá:
   - **Repository:** `TU_USUARIO/tp4-stroke-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click en **"Deploy"**. La primera vez tarda 2–4 minutos (instala dependencias).
5. Cuando termine, obtendrás una URL pública del tipo:
   `https://tp4-stroke-app-xxxx.streamlit.app` ← **esa es la URL que entregás a la cátedra.**

## Paso 4 — Verificar

- Abrí la URL en una ventana de incógnito (para confirmar que es pública).
- Cargá un paciente de prueba y comprobá que devuelve la probabilidad y los gráficos.

---

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `ModuleNotFoundError` | Falta la librería en `requirements.txt`. Agregala y la app se redeploya sola. |
| `FileNotFoundError: stroke_model.joblib` | El modelo no se subió al repo. Subilo de nuevo. |
| La app "se duerme" | En el plan gratuito se suspende por inactividad; se reactiva sola al abrir la URL (espera ~30s). |
| Error al importar `train_model` | Asegurate de que `train_model.py` esté en la raíz del repo, junto a `app.py`. |

---

## Alternativas de hosting (todas gratuitas)

- **Hugging Face Spaces** (elegí SDK = Streamlit).
- **Render** / **Railway** (requieren un `Procfile` con `web: streamlit run app.py --server.port $PORT`).

Streamlit Community Cloud es la opción más simple porque se conecta directo a GitHub.
