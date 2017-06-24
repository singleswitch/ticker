 
#!/bin/bash
# Somewhat dated. Refer to README.MD instead.
cd ../
rm -r build
python setup.py build
cp build/lib.*/audio.*  ./ 

