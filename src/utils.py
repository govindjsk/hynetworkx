import errno
import os


def get_printable_matrix(A, row_labels=None):
    A_str = ''
    A = A.toarray()
    row_labels = row_labels or ['' for i in range(A.shape[0])]
    A_str += '\n'
    if A.ndim == 1:
        A_str += str(A)
    else:
        w = max([len(str(s)) for s in A]) if len(A) > 0 else 1
        label_w = max([len(str(l)) for l in row_labels]) if len(row_labels) > 0 else 0
        A_str += ' ' * label_w + u'\u250c' + u'\u2500' * w + u'\u2510' + '\n'
        for AA, label in zip(A, row_labels):
            A_str += ' ' * (label_w - len(str(label))) + str(label)
            A_str += u'\u2502'
            A_str += ' '
            for i, AAA in enumerate(AA[:-1]):
                w1 = max([len(str(s)) for s in A[:, i]])
                A_str += str(AAA) + ' ' * (w1 - len(str(AAA)) + 1)
            w1 = max([len(str(s)) for s in A[:, -1]]) if A.shape[0] * A.shape[1] > 0 else 1
            A_str += str(AA[-1]) + ' ' * (w1 - len(str(AA[-1])))
            A_str += ' '
            A_str += u'\u2502' + '\n'
        A_str += ' ' * label_w + u'\u2514' + u'\u2500' * w + u'\u2518' + '\n'
    return A_str


def print_matrix(A, row_labels=None):
    print(get_printable_matrix(A, row_labels))


def _or(item, generator, args):
    if isinstance(item, type(None)):
        return generator(*args)
    else:
        return item


def is_iterable(item):
    if isinstance(item, str):
        return False
    try:
        list(item)
        return True
    except TypeError:
        return False


def get_data_abbr(data_name):
    data_abbr_map = {'email-Enron': 'mail',
                     'tags-math-sx': 'tag',
                     'threads-math-sx': 'thread',
                     'coauth-DBLP': 'dblp',
                     'NDC-substances': 'ndc',
                     'contact-high-school': 'contact'}
    return data_abbr_map[data_name]
    # return ''.join([x[0].upper() + x[1].lower() for x in ])


def get_data_names():
    return ['email-Enron',
            'contact-high-school',
            'NDC-substances',
            'tags-math-sx',
            'threads-math-sx',
            'coauth-DBLP']


def get_data_idx(data_name):
    return get_data_names().index(data_name)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_base_path():
    return '/home2/e1-313-15477/'
#     return '/home/govinds/repos/data/'


def get_library_path():
    return '/home2/e1-313-15477/'
#     return '/home/govinds/repos/
