from __future__ import annotations

import time
from typing import Optional
from huffman import HuffmanTree
from utils import *
import heapq

# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    if text == bytes([]):
        return {}

    frequency = {}

    if not list(text):
        return {}

    for char in text:
        if char not in frequency:
            frequency[char] = 1
        else:
            frequency[char] += 1
    return frequency


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    >>> freq = {4: 7, 56: 9, 25: 15, 8: 16, 1: 3}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(None, HuffmanTree(56), \
    HuffmanTree(None, HuffmanTree(1), HuffmanTree(4))), \
    HuffmanTree(None, HuffmanTree(25), HuffmanTree(8)))
    >>> t == result
    True
    """
    if freq_dict == {}:
        return HuffmanTree(None, None, None)
    elif len(freq_dict) == 1:

        item = freq_dict.popitem()
        freq_dict[item[0]] = item[1]

        symbol = item[0]
        if symbol == 0:
            dummy_sym = symbol + 1
        else:
            dummy_sym = symbol - 1

        actual_tree = HuffmanTree(symbol)
        dummy_tree = HuffmanTree(dummy_sym)

        freq_dict[dummy_tree.symbol] = 0

        return HuffmanTree(None, actual_tree, dummy_tree)

    else:
        sorted_dict = _sort_dict(freq_dict)

        htrees = []
        for sym, freq in sorted_dict.items():
            htrees.append((freq, HuffmanTree(sym)))

        while len(htrees) > 1:
            left_tree = htrees.pop(0)
            right_tree = htrees.pop(0)
            new_tree = HuffmanTree(None, left_tree[1], right_tree[1])

            htrees.append((left_tree[0] + right_tree[0], new_tree))
            htrees.sort()

        return htrees[0][1]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree == HuffmanTree(None, None, None):
        return {}
    return _leaf_code(tree, '')


def _leaf_code(tree: HuffmanTree, curr_code: str) -> dict[int, str]:
    """
    Return the code for a leaf with symbol <item> in this HuffmanTree.
    >>> t = HuffmanTree(None, HuffmanTree(None, HuffmanTree(56), \
    HuffmanTree(None, HuffmanTree(1), HuffmanTree(4))), \
    HuffmanTree(None, HuffmanTree(25), HuffmanTree(8)))
    >>> _leaf_code(t, '')
    {56: '00', 1: '010', 4: '011', 25: '10', 8: '11'}
    """
    codes = {}
    if tree and not tree.is_leaf():
        if tree.left:
            if tree.left.is_leaf():
                codes[tree.left.symbol] = curr_code + "0"

        if tree.right:
            if tree.right.is_leaf():
                codes[tree.right.symbol] = curr_code + "1"

        left_tree = _leaf_code(tree.left, curr_code + "0")
        right_tree = _leaf_code(tree.right, curr_code + "1")

        for sym in left_tree:
            codes[sym] = left_tree[sym]

        for sym in right_tree:
            codes[sym] = right_tree[sym]

    return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> t = HuffmanTree(None, left, right)
    >>> number_nodes(t)
    >>> t.left.number
    0
    >>> t.right.number
    1
    >>> t.number
    2
    """
    _internal_helper(tree, 0)


def _internal_helper(tree: HuffmanTree, node_num: int) -> int:
    """
    Return the number of internal nodes in this tree while numbering them.
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> t = HuffmanTree(None, left, right)
    >>> _internal_helper(t, 0)
    3
    """
    if tree.is_leaf():
        return node_num

    else:
        if tree.left:
            node_num = _internal_helper(tree.left, node_num)

        if tree.right:
            node_num = _internal_helper(tree.right, node_num)

        tree.number = node_num

        return node_num + 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    len_syms = 0
    total_frequencies = 0

    if tree == HuffmanTree(None, None, None) or freq_dict == {}:
        return 0.0

    for key, value in freq_dict.items():
        codes_dict = get_codes(tree)
        len_syms += len(codes_dict[key]) * value
        total_frequencies += value
    return len_syms / total_frequencies


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    pieces = []

    if not text or not codes:
        return bytes()

    for symbol in text:
        pieces.append(codes[symbol])

    curr_code = "".join(pieces)

    padding = (8 - len(curr_code) % 8) % 8
    curr_code += '0' * padding

    compressed_bytes = bytearray()

    for i in range(0, len(curr_code), 8):
        byte_range = curr_code[i:i + 8]
        compressed_bytes.append(int(byte_range, 2))

    return bytes(compressed_bytes)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.
    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> t = HuffmanTree(None)
    >>> tree_to_bytes(t)
    b''
    """
    if tree.is_leaf() or tree == HuffmanTree(None, None, None):
        return bytes([])  # Is it supposed to be like this??
    else:
        bytes_lst = []
        if tree.left:
            bytes_lst.extend(list(tree_to_bytes(tree.left)))

        if tree.right:
            bytes_lst.extend(list(tree_to_bytes(tree.right)))

        if tree.left:
            if tree.left.is_leaf():
                bytes_lst.append(0)
                bytes_lst.append(tree.left.symbol)
            else:
                bytes_lst.append(1)
                bytes_lst.append(tree.left.number)

        if tree.right:
            if tree.right.is_leaf():
                bytes_lst.append(0)
                bytes_lst.append(tree.right.symbol)
            else:
                bytes_lst.append(1)
                bytes_lst.append(tree.right.number)

        return bytes(bytes_lst)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    if root_index > len(node_lst) - 1 or root_index < 0:
        return HuffmanTree(None, None, None)

    tree_root = node_lst[root_index]

    if tree_root.l_type == 0:
        left_tree = HuffmanTree(tree_root.l_data, None, None)
    else:
        left_tree = generate_tree_general(node_lst, tree_root.l_data)

    if tree_root.r_type == 0:
        right_tree = HuffmanTree(tree_root.r_data, None, None)
    else:
        right_tree = generate_tree_general(node_lst, tree_root.r_data)

    return HuffmanTree(None, left_tree, right_tree)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> result = HuffmanTree(None, HuffmanTree(None, HuffmanTree(56), \
HuffmanTree(None, HuffmanTree(1), HuffmanTree(4))), \
HuffmanTree(None, HuffmanTree(25), HuffmanTree(8)))
    >>> number_nodes(result)
    >>> b = tree_to_bytes(result)
    >>> lst3 = bytes_to_nodes(b)
    >>> root_i = len(lst3) - 1
    >>> r2 = generate_tree_postorder(lst3, root_i)
    >>> res2 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(56), \
    HuffmanTree(None, HuffmanTree(1), HuffmanTree(4))), \
    HuffmanTree(None, HuffmanTree(25), HuffmanTree(8)))
    >>> r2 == res2
    True
    """
    if root_index > len(node_lst) - 1 or root_index < 0:
        return HuffmanTree(None, None, None)

    tree_root = node_lst[root_index]

    if tree_root.r_type == 0:
        right = HuffmanTree(tree_root.r_data, None, None)
    else:
        right = generate_tree_postorder(node_lst, root_index - 1)

    if tree_root.l_type == 0:
        left = HuffmanTree(tree_root.l_data, None, None)
    else:
        right_nodes = _num_nodes(right)
        left = generate_tree_postorder(node_lst, root_index - right_nodes - 1)

    return HuffmanTree(None, left, right)


def _num_nodes(tree: HuffmanTree) -> int:
    """
    Return the number of ReadNode objects that can be created from
    this HuffmanTree.
    >>> tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(10), \
    HuffmanTree(12)), \
    HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> _num_nodes(tree)
    3
    """
    if tree.is_leaf():
        return 0
    else:
        num = 1
        num += _num_nodes(tree.left)
        num += _num_nodes(tree.right)
        return num


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # Dictionary Comprehension:
    # https://www.datacamp.com/tutorial/python-dictionary-comprehension
    if tree == HuffmanTree(None, None, None) or size == 0:
        return bytes([])

    codes = get_codes(tree)
    inverted_codes = dict((c, sym) for sym, c in codes.items())
    return _decompressed_bytes_helper(text, size, inverted_codes)


def _decompressed_bytes_helper(text: bytes,
                               size: int, inv: dict[str, int]) -> bytes:
    # https://python-reference.readthedocs.io/en/latest/docs/bytearray/
    decompressed = bytearray()
    bits = ''.join(byte_to_bits(byte) for byte in text)

    code = ''
    for b in bits:
        code += b
        if code in inv:
            decompressed.append(inv[code])
            code = ''
            if len(decompressed) == size:
                break

    return bytes(decompressed)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    sorted_dict = _sort_dict(freq_dict)
    tree_lst = [tree]

    sym_lst = []
    for sym in sorted_dict:
        sym_lst.append(sym)

    if tree == HuffmanTree(None, None, None) or len(freq_dict) == 0:
        pass

    index = 0
    while len(tree_lst) > 0:
        curr_node = tree_lst.pop()
        if curr_node.left is not None:
            tree_lst.append(curr_node.left)
        if curr_node.right is not None:
            tree_lst.append(curr_node.right)
        else:
            curr_node.symbol = sym_lst[index]
            index += 1


def _sort_dict(freq_dict: dict[int, int]) -> Optional[dict[int, int]]:
    """
    Return a dictionary in ascending order based on the values in <freq_dict>.
    >>> f = {2: 6, 3: 4, 7: 5}
    >>> _sort_dict(f)
    {3: 4, 7: 5, 2: 6}
    """
    # https://stackabuse.com/how-to-sort-dictionary-by-value-in-python/
    sorted_tuples = sorted(freq_dict.items(), key=lambda data: data[1])
    sorted_dict = {}
    for sym, freq in sorted_tuples:
        sorted_dict[sym] = freq
    return sorted_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
