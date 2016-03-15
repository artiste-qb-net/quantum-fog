

class SEO_pre_reader:
    """
    This class' constructor scans an English file (a type of txt file). It
    skips all lines except the ones starting with LOOP or NEXT. By doing so,
    it collects information about LOOPs (the offset of their begin and end,
    their id number and their number of reps). This class is inherited by
    class SEO_scaner. That class scans an English file twice, first in this
    preview mode to collect info about the loops, and then it scans it more
    thoroughly, parsing each line and "using" it in some way.

    See doctring for class SEO_writer for more info about English files.

    Attributes
    ----------
    english_in : _io.TextIOWrapper
    file_prefix : str
    loop2start_offset : dict[int, int]
        a dictionary mapping loop number TO start of loop offset
    loop2tot_reps : dict[int, int]
        a dictionary mapping loop number TO total number of repetitions of
        loop
    loop_queue : list[int]
        a queue of loops labelled by their id number
    num_bits : int
        number of qubits in whole circuit
    num_lines : int
        number of lines in English file
    split_line : list[str]
        storage space for a list of string obtained by splitting a line

    """

    def __init__(self, file_prefix, num_bits):
        """
        Constructor

        Parameters
        ----------
        file_prefix : str
            file must be called file_prefix + '-' + num_bits + "eng.txt"
        num_bits : int
            total number of qubits of circuit.

        Returns
        -------

        """
        self.file_prefix = file_prefix
        self.num_bits = num_bits
        self.english_in = open(
            file_prefix + '_' + str(num_bits) + '_eng.txt', 'rt')
        self.split_line = None

        self.num_lines = 0
        self.loop2start_offset = {}
        self.loop2tot_reps = {}
        self.loop_queue = []

        self.continue_scan()

    def continue_scan(self):
        """
        if the file is still open and there is a next line, scan it; else,
        close the file.

        Returns
        -------
        None

        """
        if self.english_in.closed:
            pass

        for line in self.english_in:
            print(line)
            self.num_lines += 1
            self.scan_line(line)
        self.english_in.close()

    def scan_line(self, line):
        """
        Skips over any line that doesn't start with LOOP or NEXT. Parses
        those that do.

        Parameters
        ----------
        line : str

        Returns
        -------
        None

        """
        self.split_line = line.split()
        line_name = self.split_line[0]
        if line_name == "LOOP":
            self.scan_LOOP()
        elif line_name == "NEXT":
            self.scan_NEXT()
        else:
            pass

    def scan_LOOP(self):
        """
        Parses line starting with "LOOP". Sends parsed info to a "use"
        method.

        Returns
        -------
        None

        """
        # example:
        # LOOP 5 REPS: 2
        loop_num = int(self.split_line[1])
        reps = int(self.split_line[3])
        assert loop_num not in self.loop2tot_reps.keys(),\
            "this loop number has occurred before"
        self.loop2start_offset[loop_num] = self.english_in.tell()
        self.loop2tot_reps[loop_num] = reps
        self.loop_queue += [loop_num]

    def scan_NEXT(self):
        """
        Parses line starting with "NEXT". Sends parsed info to a "use" method.

        Returns
        -------
        None

        """
        # example:
        # NEXT 5
        loop_num = self.split_line[1]
        if not self.loop_queue:
            assert False, "unmatched NEXT"
        if loop_num == self.loop_queue[-1]:
            del self.loop_queue[-1]
        else:
            assert False, "improperly nested loops"

if __name__ == "__main__":
    print(5)