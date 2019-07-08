def print_matrix(A, row_labels=None):
    A = A.toarray()
    row_labels = row_labels or ['' for i in range(A.shape[0])]
    print()
    if A.ndim == 1:
        print(A)
    else:
        w = max([len(str(s)) for s in A]) if len(A) > 0 else 1
        label_w = max([len(str(l)) for l in row_labels]) if len(row_labels) > 0 else 0
        print(' ' * label_w + u'\u250c' + u'\u2500' * w + u'\u2510')
        for AA, label in zip(A, row_labels):
            print(' '*(label_w - len(str(label))) + str(label), end='')
            print(u'\u2502', end='')
            print(' ', end='')
            for i, AAA in enumerate(AA[:-1]):
                w1 = max([len(str(s)) for s in A[:, i]])
                print(str(AAA) + ' ' * (w1 - len(str(AAA)) + 1), end='')
            w1 = max([len(str(s)) for s in A[:, -1]]) if A.shape[0]*A.shape[1] > 0 else 1
            print(str(AA[-1]) + ' ' * (w1 - len(str(AA[-1]))), end='')
            print(' ', end='')
            print(u'\u2502')
        print(' ' * label_w + u'\u2514' + u'\u2500' * w + u'\u2518')


def _or(item, generator, args):
    if isinstance(item, type(None)):
        return generator(*args)
    else:
        return item
