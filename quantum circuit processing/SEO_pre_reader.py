

class SEO_pre_reader:
    """
    This class' constructor reads an English file (a type of txt file). It
    skips all lines except the ones starting with LOOP or NEXT. By doing so,
    it collects information about LOOPs (the offset of their begin and end,
    their id number and their number of reps). This class is inherited by
    class SEO_reader. That class reads an English file twice, first in this
    preview mode to collect info about the loops, and then it reads it more
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
            file_prefix + '_' + str(num_bits) + 'eng.txt', 'rt')
        self.split_line = None

        self.num_lines = 0
        self.loop2start_offset = {}
        self.loop2tot_reps = {}
        self.loop_queue = []

        self.continue_read_use()

    def continue_read_use(self):
        """
        if the file is still open and there is a next line, read it; else,
        close the file.

        Returns
        -------
        None

        """

        while not self.english_in.closed:
            line = self.english_in.readline()
            if not line:
                self.english_in.close()
                break
            else:
                self.num_lines += 1
                self.read_use_line(line)

    def read_use_line(self, line):
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
            self.read_use_LOOP()
        elif line_name == "NEXT":
            self.read_use_NEXT()
        else:
            pass

    def read_use_LOOP(self):
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
        self.use_LOOP(loop_num, reps)

    def use_LOOP(self, loop_num, reps):
        """
        Check that LOOP info is not illegal and store it so can do embedded
        LOOPs on next reading of file.


        Parameters
        ----------
        loop_num : int
        reps : int

        Returns
        -------
        None

        """
        assert loop_num not in self.loop2tot_reps.keys(),\
            "this loop number has occurred before"
        self.loop2start_offset[loop_num] = self.english_in.tell()
        self.loop2tot_reps[loop_num] = reps
        self.loop_queue += [loop_num]

    def read_use_NEXT(self):
        """
        Parses line starting with "NEXT". Sends parsed info to a "use" method.

        Returns
        -------
        None

        """
        # example:
        # NEXT 5
        loop_num = self.split_line[1]
        self.use_NEXT(loop_num)

    def use_NEXT(self, loop_num):
        """
        Check NEXT info is legal and if so, store it so can do embedded
        LOOPs on next reading of file.

        Parameters
        ----------
        loop_num : int

        Returns
        -------
        None

        """
        if not self.loop_queue:
            assert False, "unmatched NEXT"
        if loop_num == self.loop_queue[-1]:
            del self.loop_queue[-1]
        else:
            assert False, "improperly nested loops"

if __name__ == "__main__":
    print(5)