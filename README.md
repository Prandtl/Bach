# Bach
## How to use
To look at example results:
```zsh
make runShowExample
```
to run hogwild:
```zsh
make runHogwild
```
or to run and show:
```zsh
make runShowHogwild
```
## Files
* _example.cpp_ - simple gradient descent realization puts gradient descent path to _example.out_
* _hogwild.cpp_ - hogwild parallel SGD. Puts output to _hogwild.out_
* _ariadna.py_ - python script that takes filename as argument (*example.out* for example) and shows its contents
* _toast.cpp_ - One of PETSc examples copied from the source without changes
* _makefile_ - makefile to compile and run toast with PETSc
