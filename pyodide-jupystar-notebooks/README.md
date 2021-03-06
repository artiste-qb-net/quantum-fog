# Refit Demo_Notebook-Master.html to run in Starboard notebook

Starboard is a client side python notebook environment. It use pyodide as underlying interpreter for Python3. 

## Links 
https://starboard.gg/#python
https://github.com/gzuidhof/starboard-notebook
https://github.com/gzuidhof/starboard-notebook/blob/master/docs/static.md - See "Host your own" session

Follow these steps to install Starboard and run the notebook 

* wget https://registry.npmjs.org/starboard-notebook/-/starboard-notebook-0.7.9.tgz
* Unzip the file
* Place Demo_Notebook-Master.html in /package folder
* Place pyodide_server.py at /package folder
* Build Quantum-Fog python project and place the whl file in /package/dist folder
* Download necessary whl file from Pipy - pydot-1.4.2-py2.py3-none-any.whl, pydotplus-2.0.2-py3-none-any.whl, graphviz-0.16-py2.py3-none-any.whl
* Place them in /package/dist folder
* Start server - 'python3 pyodide_server.py'
* Access via locahost:8000/Demo_Notebook-Master.html

## Contributors

* Dekant, Henning
* Lee, Sansom
