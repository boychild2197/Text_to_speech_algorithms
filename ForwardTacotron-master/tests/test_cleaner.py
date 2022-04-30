import unittest

from utils.text.cleaners import Cleaner


class TestCleaner(unittest.TestCase):

    def test_call_happy_path(self) -> None:
        cleaner = Cleaner(cleaner_name='no_cleaners',
                          use_phonemes=True, lang='en-us')
        cleaned = cleaner('hello there!')
        self.assertEqual('həloʊ ðɛɹ!', cleaned)

        cleaned = cleaner('hello there?!.')
        self.assertEqual('həloʊ ðɛɹ?!.', cleaned)

        cleaner = Cleaner(cleaner_name='no_cleaners',
                          use_phonemes=False, lang='en-us')
        cleaned = cleaner(' Hello   there!')
        self.assertEqual('Hello there!', cleaned)

        cleaner = Cleaner(cleaner_name='english_cleaners',
                          use_phonemes=False, lang='en-us')
        cleaned = cleaner('hello there Mr. 1!')
        self.assertEqual('hello there mister one!', cleaned)
