# matrix-factorization (Repository name: Matrix-Factorization---based-Collaborative-Filtering)

matrix-factorization is a light-weight program written in python language for performing basic operations for matrix factorization. I have plans to create a python module from this repository in the future. If you want to contribute to this project, you are most welcome. 

## Requirments
### The ```matrix-factorization.py``` requires you to have ```python >= 3.7 ``` and ```numpy``` module. 

## Usage
### download the ```matrix-factorization.py``` file and put it in the same folder as your ```main.py```. After that follow the code snippet below.

```
import matrix-factorization as mf


#matrix_factorization(data-> numpy array,features-> integer,user_features-> numpy_array (optional), item_features-> numpy array (optional) )

#creating object 
d = matrix_factorization(data,features)


#train_model(learning_rate-> can be anything.(Default= 0.01) , iteration-> can be anything (Default=1000))

d.train_model(0.1,10000)

print(d.predicted_matrix())

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
