from .vertex import Vertex, generate_vertex

VERTEX_SET_DEFAULTS = {'count': 0}


def deduplicate_labels(vertices):
    id_labels_map = {}
    for v in vertices:
        if v.id in id_labels_map:
            id_labels_map[v.id].add(v.label)
        else:
            id_labels_map[v.id] = {v.label}
    if any([len(labels) > 1 for labels in id_labels_map.values()]):
        print("At least one label is unresolved. Resolving...")
    id_label_pairs = {_id: sorted(labels)[0] for _id, labels in id_labels_map.items()}
    resolved_vertices = {Vertex(_id, label) for _id, label in id_label_pairs.items()}
    return resolved_vertices


def resolve_elements(elements):
    elements = elements or []
    are_vertices = [isinstance(e, Vertex) for e in elements]
    are_ids = [isinstance(e, int) for e in elements]
    are_labels = [isinstance(e, str) for e in elements]
    if all(are_vertices):
        vertices = elements
        vertices = deduplicate_labels(vertices)
        return vertices
    elif all(are_ids):
        elements = {Vertex(e) for e in elements}
        return elements
    elif all(are_labels):
        ids = range(len(elements))
        labels = elements
        elements = {Vertex(i, e) for i, e in zip(ids, labels)}
        return elements
    else:
        raise Exception("Elements should either be vertices (Vertex),"
                        " or ids (int), or labels (str).")


class VertexSet(set):
    def __init__(self, elements=()):
        elements = resolve_elements(elements)
        super(VertexSet, self).__init__(elements)

    def __repr__(self):
        return 'VertexSet({})'.format(',\n\t\t'.join(map(str, sorted(self))))

    # def __hash__(self):
    #     return frozenset.__hash__(self.elements)

    @property
    def __summary__(self):
        return {v.__summary__ for v in self}

    # @classmethod
    # def _wrap_methods(cls, names):
    #     def wrap_method_closure(name):
    #         def inner(self, *args):
    #             result = getattr(super(cls, self), name)(*args)
    #             if isinstance(result, set):
    #                 result = cls(result)
    #             return result
    #
    #         inner.fn_name = name
    #         setattr(cls, name, inner)
    #
    #     for name in names:
    #         wrap_method_closure(name)


# VertexSet._wrap_methods(['__ror__', 'difference_update', '__isub__',
#                          'symmetric_difference', '__rsub__', '__and__', '__rand__', 'intersection',
#                          'difference', '__iand__', 'union', '__ixor__',
#                          'symmetric_difference_update', '__or__', 'copy', '__rxor__',
#                          'intersection_update', '__xor__', '__ior__', '__sub__',
#                          ])

#
# class VertexSet(set):
#     def __init__(self, elements=None):
#         self.set = elements
#
#     @property
#     def set(self):
#         return self._set
#
#     @set.setter
#     def set(self, elements):
#         elements = elements or set()
#         are_vertices = [isinstance(e, Vertex) for e in elements]
#         are_ids = [isinstance(e, int) for e in elements]
#         are_labels = [isinstance(e, str) for e in elements]
#         if all(are_vertices):
#             self._set = {e for e in elements}
#         elif all(are_ids):
#             elements = {Vertex(e) for e in elements}
#             self._set = elements
#         elif all(are_labels):
#             ids = range(len(elements))
#             labels = elements
#             elements = {Vertex(i, e) for i, e in zip(ids, labels)}
#             self._set = elements
#         else:
#             raise Exception("Elements should either be vertices (Vertex), or ids (int), or labels (str).")
#
#     @property
#     def cardinality(self):
#         return len(self.set)
#
#     def __repr__(self):
#         return 'VertexSet{}'.format(self.set if self.cardinality else '{}')
#
#     def __eq__(self, other):
#         return self.__class__ == other.__class__ and self.set == other.set
#
#     def __contains__(self, item):
#         return item in self.set


def generate_monotonic_vertex_set(count: int = None,
                                  label_prefix: str = None):
    V = generate_vertex_set(count, (0, count), label_prefix)
    return V


def generate_vertex_set(count: int = None,
                        id_range: tuple = None,
                        label_prefix: str = None,
                        reserved_ids: set = None,
                        reserved_vertex_sets: set = None) -> VertexSet:
    """
    Generates a set of random vertices.
    :param count: Number of vertices to generate.
    :param id_range: Tuple specifying upper and lower limits (both included) of where vertex ids should lie.
    :param label_prefix: Prefix string to attach to the vertex ids to form vertex labels.
    :param reserved_ids: Set (or iterator) of ids that are already reserved.
    :param reserved_vertex_sets:
    :return: A set of vertices (a VertexSet object).
    """
    count = count or VERTEX_SET_DEFAULTS['count']
    reserved_vertex_sets = set(reserved_vertex_sets or [])
    V = VertexSet()
    while len(V) < count:
        v = generate_vertex(id_range, label_prefix, reserved_ids)
        reserved_ids = set(reserved_ids or [])
        reserved_ids.add(v.id)
        V.add(v)
    if V in reserved_vertex_sets:
        V = generate_vertex_set(count, id_range, label_prefix, reserved_ids, reserved_vertex_sets)
    return V


def main():
    pass


if __name__ == '__main__':
    main()
