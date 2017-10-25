import timeit

cy = timeit.timeit('example_cy.example(10000)', setup = 'import example_cy', number = 100)
py = timeit.timeit('example_py.example(10000)', setup = 'import example_py', number = 100)

print('Cython is {}x faster!'.format(py/cy))