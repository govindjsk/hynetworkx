from random import choice
VERTEX_DEFAULTS = {'id_range': (0, 100),
                   'label_prefix': 'vertex'}


class Vertex(tuple):
    def __new__(cls, _id: int = 0, label: str = None):
        assert isinstance(_id, int), 'Vertex id has to be an integer.'
        assert isinstance(label or '', str), 'Vertex label has to be either None or a string.'
        return super(Vertex, cls).__new__(Vertex, (_id, label))

    @property
    def id(self):
        return self[0]

    @property
    def label(self):
        return self[1]

    @property
    def __summary__(self):
        return self.id

    def __repr__(self):
        return 'Vertex({}, {})'.format(self.id, self.label)


def generate_vertex(id_range=None, label_prefix=None, reserved_ids=None):
    """
    Generates a random vertex.
    :param id_range: Tuple specifying upper and lower limits (both included) of where vertex id should lie.
    :param label_prefix: Prefix string to attach to the vertex id to form vertex label.
    :param reserved_ids: Set (or iterator) of ids that are already reserved.
    :return: A vertex (a Vertex object).
    """
    id_range = id_range or VERTEX_DEFAULTS['id_range']
    label_prefix = label_prefix or VERTEX_DEFAULTS['label_prefix']
    reserved_ids = set(reserved_ids or [])
    # TODO: Optimize this. Might take a lot of resources.
    total_ids = set(range(*id_range))
    available_ids = total_ids.difference(reserved_ids)
    capacity = len(available_ids)
    if capacity == 0:
        raise Exception('Not enough vertex ids.')
    _id = choice(list(available_ids))
    label = label_prefix + str(_id)
    return Vertex(_id, label)


def main():
    pass


if __name__ == '__main__':
    main()
