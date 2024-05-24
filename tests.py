import unittest
from Bio import SeqIO
from Bio.Seq import Seq

###############################################
#################Functions_to_test#############
###############################################
def seq_finder(file):
    
    sequences = []
    for sequence in SeqIO.parse(file, "fasta"):
        sequences.append((sequence.id, str(sequence.seq)))  
    return sequences


def fragmentor(sequence, id='', maxseq=75, overlap=10, max_gap=5):
    fragments = []
    step_size = maxseq - overlap
    if len(sequence) <= maxseq:
        return fragments
    else:
        for i in range(0, len(sequence), step_size):
            fragment = sequence[i:i + maxseq]
            if len(fragment) < maxseq:
                fragment += '-' * (maxseq - len(fragment))
            num_ = fragment.count('-')
            if ( num_ * 100 / maxseq) < max_gap:
                fragments.append((fragment,id))
    return fragments



class TestSeqFinder(unittest.TestCase):
    def test_seq_finder(self):
        # Test input file path
        path_test = "/home/aladdine_lekchiri/Téléchargements/DB/fasta_test.fa"
        
        # Expected output
        expected = [
            ("chr1", "AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGCTATCGATCGTAGCTAGCTGACTGACTGACTGATGCTAGCTAGCTAGCTAGCTAGCTAGCTGATCGATCGATCGACTGACTGACTGATG"),
            ("chr2", "AGTCCGATAGCTACGATCGATCGATCGATCGATA"),
            ("chr3", "AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGAGC")
        ]
        
        result = seq_finder(path_test)
        
        # Checking if the result matches the expected output
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
# Test cases
class TestFragmentor(unittest.TestCase):
    def test_basic_case(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = []
        self.assertEqual(fragmentor(seq, maxseq=75, overlap=10, max_gap=5), expected)
    
    def test_exact_match(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = []
        self.assertEqual(fragmentor(seq, maxseq=80, overlap=0, max_gap=5), expected)

    def test_less_than_fragment_size(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = [("ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACT","")]
        self.assertEqual(fragmentor(seq, maxseq=75, overlap=10, max_gap=5), expected)

    def test_overlap_and_gaps(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = []
        self.assertEqual(fragmentor(seq, maxseq=68, overlap=10, max_gap=50), expected)



# Run 
if __name__ == "__main__":
    unittest.main()
