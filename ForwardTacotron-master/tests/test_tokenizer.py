import unittest

from utils.text.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def test_call_happy_path(self) -> None:
        tokenizer = Tokenizer()
        tokens = tokenizer('_ abc{')
        self.assertEqual([0, 10, 36, 52, 57], tokens)

        decoded = tokenizer.decode(tokens)
        self.assertEqual('_ abc', decoded)