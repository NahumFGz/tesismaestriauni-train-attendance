## 📘 Comandos útiles de Poetry

### 🔧 Configuración inicial

```bash
python -m venv venv_miproyecto
# Crea tu entorno virtual personalizado

source venv_miproyecto/bin/activate
# Activa el entorno virtual (Linux/macOS)
# venv_miproyecto\Scripts\activate  # En Windows

poetry config virtualenvs.create false
# Usa el venv activado manualmente (Poetry no crea uno propio)

poetry init
# Inicializa pyproject.toml con preguntas interactivas
```

---

### 📦 Gestión de dependencias

```bash
poetry add <paquete>
# Añade una dependencia principal y actualiza pyproject.toml + poetry.lock

poetry add --dev <paquete>
# Añade una dependencia de desarrollo (ej. pytest, black)

poetry remove <paquete>
# Elimina una dependencia

poetry update
# Actualiza todas las dependencias y actualiza el poetry.lock

poetry update <paquete>
# Actualiza solo un paquete específico
```

---

### 🧪 Uso del entorno virtual

```bash
poetry install
# Instala todas las dependencias definidas en pyproject.toml y poetry.lock
# (usa el entorno virtual activo si virtualenvs.create = false)

poetry run python <script.py>
# Ejecuta un script dentro del entorno gestionado por Poetry

poetry shell
# Abre una shell dentro del entorno virtual (solo si Poetry lo creó)
```

---

### 🔍 Inspección y depuración

```bash
poetry show
# Lista los paquetes instalados

poetry show --tree
# Muestra árbol completo de dependencias y transitivas

poetry check
# Verifica que pyproject.toml esté bien formado
```

---

### 🧹 Otros comandos útiles

```bash
poetry lock
# Regenera el archivo poetry.lock sin instalar nada

poetry version
# Muestra o cambia la versión del proyecto

poetry env info
# Muestra información del entorno virtual
```

---

### 📝 Tips adicionales

- Para desactivar la creacion de .env por defecto `poetry config virtualenvs.create false`

- Mostrar el arbol de dependencias `poetry show --tree`

- Usa `python -m venv <nombre>` para crear un entorno virtual con nombre personalizado.

- Añade `.venv/`, `.venv_miproyecto/` o similares a tu `.gitignore`.

- Si borras `poetry.lock`, se regenerará con nuevas versiones al hacer `poetry install`.

- Si **no necesitas instalar tu propio código como paquete**, desactiva el modo paquete en `pyproject.toml`:

  ```toml
  [tool.poetry]
  package-mode = false
  ```

- O bien, usa `poetry install --no-root` para instalar solo las dependencias sin intentar instalar tu proyecto como módulo.
