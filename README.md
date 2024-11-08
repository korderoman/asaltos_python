activar el entorno: source asaltos_env/bin/activate
ejecutar la aplicacón: python -m src.main
Si tienes error de memoria, es decir que la instalación se rompe por falta de memoria, 
haz que la instalación sea en segundo plano: pip install --no-cache-dir -r requirements.txt

Si tienes un error de este tipo: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
debes de instalar: sudo apt install -y libgl1-mesa-glx