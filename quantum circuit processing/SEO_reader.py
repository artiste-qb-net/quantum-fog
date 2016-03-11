from QLib.src.Controls import *
from QLib.src.SEO_pre_reader import *


class SEO_reader(SEO_pre_reader):
    """
    This class inherits from the class SEO_pre_reader. It's an abstract
    class because it has a bunch of use_ methods that must be overridden by
    a child class. This class reads each line of an English file, parses it,
    and sends the info obtained to a use_ method for further processing. One
    very important child of this class is SEO_simulator which uses each line
    of the English file to evolve by one further step a quantum state vector.

    See the docstring for the class SEO_reader for more info about
    English files.


    Attributes
    ----------

    num_ops : int
        number of operations (lines of English file rerun n times count n)
    loop2cur_rep : dict[int, int]
        a dictionary mapping loop number TO current repetition

    english_in : _io.TextIOWrapper
    file_prefix : str
    loop2start_offset : dict[int, int]
    loop2tot_reps : dict[int, int]
    loop_queue : list[int]
    num_bits : int
    num_lines : int
    split_line : list[str]

    """
    
    def __init__(self, file_prefix, num_bits):
        """
        Constructor

        Parameters
        ----------
        file_prefix : str
        num_bits : int

        Returns
        -------

        """
        SEO_pre_reader.__init__(self, file_prefix, num_bits)

        self.english_in = open(
            file_prefix + '_' + str(num_bits) + 'eng.txt', 'rt')

        self.loop2cur_rep = {loop_num: -1 for
                              loop_num in self.loop2tot_reps.keys()}
        self.num_ops = 0

        self.continue_read_use()

        self.write_log()

    def write_log(self):
        """
        Write a log file and print info on console too.

        Returns
        -------
        None

        """
        log_out = open(
            self.file_prefix + '_' + str(self.num_bits) + 'log.txt', 'wt')

        s = "Number of lines in file = " + str(self.num_lines)
        log_out.write(s)
        print(s)

        s = "Number of Elem. Ops = " + str(self.num_ops)
        log_out.write(s)
        print(s)

        log_out.close()

    def read_use_line(self, line):
        """
        Analyze the inputted line. Send to different read_use methods
        labelled by first four letters of line.


        Parameters
        ----------
        line : str

        Returns
        -------
        None

        """
        self.split_line = line.split()
        line_name = self.split_line[0]
        self.num_ops += 1
        if line_name == 'NOTA':
            # don't count NOTA as operation
            self.num_ops -= 1
            self.use_NOTA(line[4:])
        elif line_name == "LOOP":
            # don't count LOOP as operation
            self.num_ops -= 1
            self.read_use_LOOP()
        elif line_name == "NEXT":
            # don't count NEXT as operation
            self.num_ops -= 1
            self.read_use_NEXT()
        elif line_name == "SWAP":
            self.read_use_SWAP()
        elif line_name == "PHAS":
            self.read_use_PHAS()
        elif line_name == "P0PH":
            self.read_use_P_phase_factor(0)
        elif line_name == "P1PH":
            self.read_use_P_phase_factor(1)
        elif line_name == "SIGX":
            self.read_use_SIG(1)
        elif line_name == "SIGY":
            self.read_use_SIG(2)
        elif line_name == "SIGZ":
            self.read_use_SIG(3)
        elif line_name == "HAD2":
            self.read_use_HAD2()
        elif line_name == "ROTX":
            self.read_use_ROT(1)
        elif line_name == "ROTY":
            self.read_use_ROT(2)
        elif line_name == "ROTZ":
            self.read_use_ROT(3)
        elif line_name == "ROTN":
            self.read_use_ROTN()
        else:
            assert False, \
                "reading an unsupported line kind: " + line_name

    def use_NOTA(self, bla_str):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        bla_str : str

        Returns
        -------
        None

        """
        assert False
        
    def read_use_LOOP(self):
        """
        Collect useful info from LOOP split_line and forward it to abstract
        use_ method.

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
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        loop_num : int
        reps : int

        Returns
        -------
        None

        """
        self.loop2cur_rep[loop_num] += 1
        self.continue_read_use()

    def read_use_NEXT(self):
        """
        Collect useful info from NEXT split_line and forward it to abstract
        use_ method.

        Returns
        -------
        None

        """
        # example:
        # NEXT 5

        loop_num = int(self.split_line[1])
        self.use_NEXT(loop_num)

    def use_NEXT(self, loop_num):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        loop_num : int

        Returns
        -------
        None

        """
        if self.loop2cur_rep[loop_num] < self.loop2tot_reps[loop_num]:
            self.english_in.seek(self.loop2start_offset)
        else:
            self.loop2cur_rep[loop_num] = -1
        self.continue_read_use()

    def read_use_SWAP(self):
        """
        Collect useful info from SWAP split_line and forward it to abstract
        use_ method.

        Returns
        -------
        None

        """
        # example:
        # SWAP 1 0 IF 3F 2T

        bit1 = int(self.split_line[1])
        bit2 = int(self.split_line[2])
        controls = self.read_TF_controls(self.split_line[4:])
        self.use_SWAP(bit1, bit2, controls)

    def use_SWAP(self, bit1, bit2, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        bit1 : int
        bit2 : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False
        
    def read_use_PHAS(self):
        """
        Collect useful info from PHAS split_line and forward it to abstract
        use_ method.

        Returns
        -------
        None

        """
        # example:
        # PHAS 42.7 AT 3 IF 3F 2T

        angle_degs = float(self.split_line[1])
        tar_bit_pos = int(self.split_line[3])
        controls = self.read_TF_controls(self.split_line[5:])
        self.use_PHAS(angle_degs, tar_bit_pos, controls)

    def use_PHAS(self, angle_degs, tar_bit_pos, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        angle_degs : float
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False
        
    def read_use_P_phase_factor(self, projection_bit):
        """
        Collect useful info from P0PH or P1PH split_line and forward it to
        abstract use_ method.

        Parameters
        ----------
        projection_bit : int

        Returns
        -------
        None

        """
        # example:
        # P0PH 42.7 AT 3 IF 2T
        # P1PH 42.7 AT 3 IF 2T

        angle_degs = float(self.split_line[1])
        tar_bit_pos = int(self.split_line[3])
        controls = self.read_TF_controls(self.split_line[5:])
        assert projection_bit in [0, 1]
        self.use_P_PH(projection_bit,
                          angle_degs, tar_bit_pos, controls)

    def use_P_PH(self, projection_bit,
                angle_degs, tar_bit_pos, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        projection_bit : int
        angle_degs : float
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False

    def read_use_SIG(self, direction):
        """
        Collect useful info from SIGX, SIGY, or SIGZ split_line and forward
        it to abstract use_ method.

        Parameters
        ----------
        direction : int

        Returns
        -------
        None

        """
        # example:
        # SIGX AT 1 IF 3F 2T
        # SIGY AT 1 IF 3F 2T
        # SIGZ AT 1 IF 3F 2T

        tar_bit_pos = int(self.split_line[2])
        controls = self.read_TF_controls(self.split_line[4:])
        assert direction in [1, 2, 3]
        self.use_SIG(direction, tar_bit_pos, controls)

    def use_SIG(self, direction, tar_bit_pos, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        direction : int
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False

    def read_use_HAD2(self):
        """
        Collect useful info from HAD2 split_line and forward it to abstract
        use_ method.

        Returns
        -------
        None

        """
        # example:
        # HAD2 AT 1 IF 3F 2T

        tar_bit_pos = int(self.split_line[2])
        controls = self.read_TF_controls(self.split_line[4:])
        self.use_HAD2(tar_bit_pos, controls)

    def use_HAD2(self, tar_bit_pos, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False

    def read_use_ROT(self, direction):
        """
        Collect useful info from ROTX, ROTY, or ROTZ split_line and forward
        it to abstract use_ method.

        Parameters
        ----------
        direction : int

        Returns
        -------
        None

        """
        # example:
        # ROTX 42.7 AT 3 IF 3F 2T
        # ROTY 42.7 AT 3 IF 3F 2T
        # ROTZ 42.7 AT 3 IF 3F 2T

        angle_degs = float(self.split_line[1])
        tar_bit_pos = int(self.split_line[3])
        controls = self.read_TF_controls(self.split_line[5:])
        self.use_ROT(direction,
                         angle_degs, tar_bit_pos, controls)

    def use_ROT(self, direction,
            angle_degs, tar_bit_pos, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        direction : int
        angle_degs : float
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False

    def read_use_ROTN(self):
        """
        Collect useful info from ROTN split_line and forward it to abstract
        use_ method.

        Returns
        -------
        None

        """
        # example:
        # ROTN 42.7 30.2 78.5 AT 3 IF 3F 2T

        angle_x_degs = float(self.split_line[1])
        angle_y_degs = float(self.split_line[2])
        angle_z_degs = float(self.split_line[3])
        tar_bit_pos = int(self.split_line[5])
        controls = self.read_TF_controls(self.split_line[7:])
        self.use_ROTN(angle_x_degs, angle_y_degs, angle_z_degs,
                         tar_bit_pos, controls)

    def use_ROTN(self, angle_x_degs, angle_y_degs, angle_z_degs,
                tar_bit_pos, controls):
        """
        Abstract use_ method that must be overridden by child class.

        Parameters
        ----------
        angle_x_degs : float
        angle_y_degs : float
        angle_z_degs : float
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        assert False

    def read_TF_controls(self, tokens):
        """
        Given a list of tokens of the form: an int followed by either T or F,
        construct a T/F control out of it.

        Parameters
        ----------
        tokens : list[str]

        Returns
        -------
        Controls

        """
        # safe to use when no "IF"
        # when no "IF", will return controls with _numControls=0
        controls = Controls(self.num_bits)
        if tokens:
            for t in tokens:
                assert t[-1] in ['T', 'F']
                controls.set_control(int(t[:-1]),
                                     True if t[-1] == 'T' else False)

        controls.refresh_lists()
        return controls

if __name__ == "__main__":
    print(5)