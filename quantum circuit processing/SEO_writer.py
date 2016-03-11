from QLib.src.CktEmbedder import *
from QLib.src.Controls import *
from QLib.src.OneBitGates import *
import re


class SEO_writer:
    """
    The constructor of this class opens an English and a Picture file. Every
    other method of the class writes, each time it is called, a single line
    in each of those 2 files.

    Note SEO stands for Sequence of Elementary Operations.

    So what are English and Picture files?

    We use 3 types of files to characterize a single quantum circuit (in the
    gate model): (1) an English file (2) a Picture file (3) a Log file.

    Log files are written by class SEO_reader, whereas English and Picture
    files are written by this class, SEO_writer.

    A log file just contains useful information like the number of lines of
    the English file (same as that of Picture file) and their number of
    operations.

    The lines of an English and Picture file are in 1-1 correspondence,
    each line representing a single operation (e.g., a multi-controlled one
    qubit gate or a multi-controlled 2 qubit swap).

    The English file contains complete info about the operation specified by
    each line. A Picture file contains only partial info about each line in
    the form of an ascii picture.

    In English and Picture files, time flows downward.

    The class SEO_writer allows the bool option zero_bit_first. When this
    argument is set to True (resp., False), the Picture file shows the zero
    qubit first (resp., last), and the remaining qubits in consecutive
    order. Picture files written with zero bit first ( resp., last) are
    labelled prefix + '_ZFpict.text' (reps., prefix + '_ZLpict.txt').

    See the following and earlier arXiv papers by R.R.Tucci for more info on
    English and Picture files.

        http://arxiv.org/abs/1004.2205
        "Quibbs, a Code Generator for Quantum Gibbs Sampling"

    The following images from the above paper are stored in the same folder
    as this python file.

        pic_file_examples.png
        eng_file_examples.png

    These 2 images give examples of lines in analytic/Picture/English formats.

    Attributes
    ----------
    emb : CktEmbedder
    english_out : _io.TextIOWrapper
    picture_out : _io.TextIOWrapper
    file_prefix : str
    line_counter : int
    num_bits : int

    zero_bit_first : bool

    """

    def __init__(self, file_prefix, num_bits, zero_bit_first):
        """
        Constructor

        Parameters
        ----------
        file_prefix : str
        num_bits : int
        zero_bit_first : bool

        Returns
        -------

        """
        self.line_counter = 0
        self.file_prefix = file_prefix
        self.english_out = open(
            file_prefix + '_' + str(num_bits) + 'eng.txt', 'wt')
        self.picture_out = open(
            file_prefix + '_' + str(num_bits) +
            ('ZF' if zero_bit_first else 'ZL') + 'pic.txt', 'wt')
        self.emb = None
        self.num_bits = num_bits
        self.zero_bit_first = zero_bit_first

        assert num_bits >= 2, "number of bits must be >= 2"

    def close_files(self):
        """
        Closes English and Picture files that were opened by the constructor.

        Returns
        -------
        None

        """
        self.english_out.close()
        self.picture_out.close()

    def write_NOTA(self, bla_str):
        """
        Write a 'NOTA' line in eng & pic files. As the name implies, a NOTA
        is just a note or comment such as "I love you Mary". It is not a gate.

        Parameters
        ----------
        bla_str : str

        Returns
        -------
        None

        """
        s = "NOTA\t" + bla_str
        self.english_out.write(s)
        self.picture_out.write(s)

    def write_LOOP(self, loop_num, reps):
        """
        Write a 'LOOP' line in eng & pic files. The gates between a LOOP
        line and its partner NEXT line are supposed to be repeated a number
        of times called reps.

        Parameters
        ----------
        loop_num : int
        reps : int

        Returns
        -------
        None

        """
        s = "LOOP\t" + str(loop_num) + "\tREPS:\t" + str(reps)
        self.english_out.write(s)
        self.picture_out.write(s)

    def write_NEXT(self, loop_num):
        """
        Write a 'NEXT' line in eng & pic files.

        Parameters
        ----------
        loop_num : int

        Returns
        -------
        None

        """
        s = "NEXT\t" + str(loop_num)
        self.english_out.write(s)
        self.picture_out.write(s)
        
    def write_pic_line(self, pic_line):
        """
        Writes a line in the Picture file only using either the ZF or ZL
        conventions. pic_line is originally written as ZL format,
        so no change if ZL option chosen but reverse order of gates if ZF
        option chosen.

        Parameters
        ----------
        pic_line : str

        Returns
        -------
        None

        """
        # example:
        # Ry--R---<--->---Rz
        if self.zero_bit_first:
            nodes = re.split('(--|---)', pic_line)
            nodes = list(reversed(nodes))
            for nd in nodes:
                if nd == '<':
                    nd = '>'
                elif nd == '>':
                    nd = '<'
            dos = '--'
            tres = '---'
            new_line = ''
            k = 0
            for nd in nodes:
                if k < len(nodes) - 1:
                    if len(nd) == 1:
                        new_line += tres
                    elif len(nd) == 2:
                        new_line += dos
                k += 1
        else:
            pass

        self.picture_out.write(pic_line)

    def write_controlled_bit_swap(self, bit1, bit2, trols):
        """
        Writes a line in eng & pic files for a 'SWAP' with >= 0 controls. 

        Parameters
        ----------
        bit1 : int
        bit2 : int
        trols : Controls

        Returns
        -------
        None

        """

        # preamble, same for all 3 controlled gate methods
        self.line_counter += 1
        assert not self.english_out.closed
        assert not self.picture_out.closed

        num_bits_bef = self.emb.num_bits_bef
        num_bits_aft = self.emb.num_bits_aft
        # aft_tar_bit_pos = self.emb.aft(tar_bit_pos)

        aft_trols = trols.new_embedded_self(self.emb)

        # number of controls may be zero
        num_controls = len(aft_trols.bit_pos)
        # end of preamble

        assert bit1 != bit2, "swapped bits must be different"
        assert -1 < bit1 < num_bits_bef
        assert -1 < bit2 < num_bits_bef
        x = [self.emb.aft(bit1), self.emb.aft(bit2)]
        big = max(x)
        small = min(x)

        # english file
        self.english_out.write("SWAP\t" + str(big) + "\t" + str(small))
        self.english_out.write("\tIF\t" if num_controls != 0 else "\n")

        # list bit-positions in decreasing order
        for c in range(num_controls):
            self.english_out.write(str(aft_trols.bit_pos[c]) +
                "T" if aft_trols.kinds[c] == True else "F" +
                "\n" if c == num_controls - 1 else "\t")

        # picture file
        pic_line = ""
        biggest = big
        smallest = small
        if num_controls != 0:
            biggest = max(aft_trols.bit_pos[0], big)
            smallest = min(aft_trols.bit_pos[num_controls-1], small)

        # k a bit position
        for k in range(num_bits_aft-1, biggest, -1):
            pic_line += "|   "

        c_int = 0
        for k in range(biggest, smallest-1, -1):
            is_big = (k == big)
            is_small = (k == small)
            is_control = False
            control_kind = False
            tres = ":" if (k == smallest) else "---"
            # dos = ":" if (k == smallest) else "--"

            for c in range(c_int, num_controls, +1):
                if k == aft_trols.bit_pos[c]:
                    is_control = True
                    control_kind = aft_trols.kinds[c]
                    c_int += 1
                    break

            if is_control:
                pic_line += "<" + tres
            else:  # control not found
                if is_big:
                    pic_line += "<" + tres
                elif is_small:
                    pic_line += ">" + tres
                else:
                    pic_line += "+" + tres

        for k in range(smallest-1, -1, -1):
            pic_line += "|   "
            
        self.write_pic_line(pic_line)
        self.picture_out.write("\n")

    def write_controlled_one_bit_gate(
            self, tar_bit_pos, trols, one_bit_gate_fun, fun_arg_list):
        """
        Writes a line in eng & pic files for a one bit gate (from class 
        OneBitGates) with >= 0 controls. 

        Parameters
        ----------
        tar_bit_pos : int
        trols : Controls
        one_bit_gate_fun : Any->np.ndarray
        fun_arg_list : list[]

        Returns
        -------
        None

        """

        # preamble, same for all 3 controlled gate methods
        self.line_counter += 1
        assert not self.english_out.closed
        assert not self.picture_out.closed

        # num_bits_bef = self.emb.num_bits_bef
        num_bits_aft = self.emb.num_bits_aft
        aft_tar_bit_pos = self.emb.aft(tar_bit_pos)

        aft_trols = trols.new_embedded_self(self.emb)

        # number of controls may be zero
        num_controls = len(aft_trols.bit_pos)
        # end of preamble

        assert tar_bit_pos not in aft_trols.bit_pos,\
            "target bit cannot be a control bit"

        # english file
        if one_bit_gate_fun == OneBitGates.one_bit_phase_fac:
            self.english_out.write("PHAS\t" +
            str(fun_arg_list[0]*180/np.pi))
        elif one_bit_gate_fun == OneBitGates.one_bit_P_0_phase_fac:
            self.english_out.write("P0PH\t" +
                str(fun_arg_list[0]*180/np.pi))
        elif one_bit_gate_fun == OneBitGates.one_bit_P_1_phase_fac:
            self.english_out.write("P1PH\t" +
                str(fun_arg_list[0]*180/np.pi))
        elif one_bit_gate_fun == OneBitGates.sigx:
            self.english_out.write("SIGX")
        elif one_bit_gate_fun == OneBitGates.sigy:
            self.english_out.write("SIGY")
        elif one_bit_gate_fun == OneBitGates.sigz:
            self.english_out.write("SIGZ")
        elif one_bit_gate_fun == OneBitGates.had2:
            self.english_out.write("HAD2")
        elif one_bit_gate_fun == OneBitGates.one_bit_rot_ax:
            ang_rads = fun_arg_list[0]
            axis = fun_arg_list[1]
            if axis == 1:
                self.english_out.write("ROTX\t" + str(ang_rads*180/np.pi))
            elif axis == 2:
                self.english_out.write("ROTY\t" + str(ang_rads*180/np.pi))
            elif axis == 3:
                self.english_out.write("ROTZ\t" + str(ang_rads*180/np.pi))
            else:
                assert False
        elif one_bit_gate_fun == OneBitGates.one_bit_rot:
            x_degs = fun_arg_list[0]*180/np.pi
            y_degs = fun_arg_list[1]*180/np.pi
            z_degs = fun_arg_list[2]*180/np.pi
            self.english_out.write("ROTN\t" +
                str(x_degs) + "\t" + str(y_degs) + "\t" + str(z_degs))
        else:
            assert False, "writing an unsupported controlled gate"

        self.english_out.write("\tAT\t" + str(aft_tar_bit_pos) +
            "\tIF\t" if num_controls != 0 else "\n")

        # list bit-positions in decreasing order
        for c in range(num_controls):
            self.english_out.write(str(aft_trols.bit_pos[c]) +
                "T" if aft_trols.kinds[c] == True else "F" +
                "\n" if c == num_controls - 1 else "\t")

        # picture file
        pic_line = ""
        biggest = aft_tar_bit_pos
        smallest = aft_tar_bit_pos
        if num_controls != 0:
            biggest = max(aft_trols.bit_pos[0], aft_tar_bit_pos)
            smallest = min(aft_trols.bit_pos[num_controls-1], aft_tar_bit_pos)

        # k a bit position
        for k in range(num_bits_aft-1, biggest, -1):
            pic_line += "|   "

        c_int = 0
        for k in range(biggest, smallest-1, -1):
            is_target = (k == tar_bit_pos)
            is_control = False
            control_kind = False
            tres = ":" if (k == smallest) else "---"
            dos = ":" if (k == smallest) else "--"

            # c_int starts at last value
            for c in range(c_int, num_controls, +1):
                if k == aft_trols.bit_pos[c]:
                    is_control = True
                    control_kind = aft_trols.kinds[c]
                    c_int += 1
                    break

            if is_control:
                pic_line += "@" if control_kind else "O" + tres
            else:  # is not control
                if not is_target:  # is not control or target
                    pic_line += "+" + tres
                else:  # is target
                    if one_bit_gate_fun == OneBitGates.one_bit_phase_fac:
                        pic_line += "Ph" + dos
                    elif one_bit_gate_fun == OneBitGates.one_bit_P_0_phase_fac:
                        pic_line += "OP" + dos
                    elif one_bit_gate_fun == OneBitGates.one_bit_P_1_phase_fac:
                        pic_line += "@P" + dos
                    elif one_bit_gate_fun == OneBitGates.sigx:
                        pic_line += "X" + tres
                    elif one_bit_gate_fun == OneBitGates.sigy:
                        pic_line += "Y" + tres
                    elif one_bit_gate_fun == OneBitGates.sigz:
                        pic_line += "Z" + tres
                    elif one_bit_gate_fun == OneBitGates.had2:
                        pic_line += "H" + tres
                    elif one_bit_gate_fun == OneBitGates.one_bit_rot_ax:
                        ang_rads = fun_arg_list[0]
                        axis = fun_arg_list[1]
                        if axis == 1:
                            pic_line += "Rx" + dos
                        elif axis == 2:
                            pic_line += "Ry" + dos
                        elif axis == 3:
                            pic_line += "Rz" + dos
                        else:
                            assert False
                    elif one_bit_gate_fun == OneBitGates.one_bit_rot:
                        pic_line += "R" + tres
                    else:
                        assert False, "writing an unsupported controlled gate"

        for k in range(smallest-1, -1, -1):
            pic_line += "|   "
            
        self.write_pic_line(pic_line)
        self.picture_out.write("\n")

    def write_controlled_multiplexor_gate(self, tar_bit_pos,
            trols, with_minus, rad_angles):
        """

        Writes a line in eng & pic files for a multiplexor 'MP_Y' with
        >= 0 controls.

        The definition of multiplexors and how they are specified in
        both English and Picture files is described in the references given
        in the docstring of this class.

        Parameters
        ----------
        tar_bit_pos : int
        trols : Controls
        with_minus : bool
        rad_angles : list[float]

        Returns
        -------
        None

        """

        # preamble, same for all 3 controlled gate methods
        self.line_counter += 1
        assert not self.english_out.closed
        assert not self.picture_out.closed

        # num_bits_bef = self.emb.num_bits_bef
        num_bits_aft = self.emb.num_bits_aft
        aft_tar_bit_pos = self.emb.aft(tar_bit_pos)

        aft_trols = trols.new_embedded_self(self.emb)

        # number of controls may be zero
        num_controls = len(aft_trols.bit_pos)
        # end of preamble
        
        assert tar_bit_pos not in aft_trols.bit_pos,\
            "target bit cannot be a control bit"

        num_int_controls = aft_trols.get_num_int_controls()
        assert num_int_controls != 0, \
            "multiplexor with no half-moon controls"
        num_angles = len(rad_angles)
        assert num_angles == (1 << num_int_controls),\
            "wrong number of multiplexor angles"
        
        # english file
        self.english_out.write(
            "MP_Y\tAT\t" + str(aft_tar_bit_pos) + "\tIF\t")
        
        # list bit-positions in decreasing order
        for c in range(num_controls):
            x = aft_trols.kinds[c]
            kind_str = ""
            if isinstance(x, int):
                kind_str = "(" + str(x)
            elif not x:
                kind_str = "F"
            elif x:
                kind_str = "T"
            self.english_out.write(
                str(aft_trols.bit_pos[c]) + kind_str + "\t")
            # use BY to indicate end of controls
            self.english_out.write("\tBY\t")
            for k in range(num_angles):
                self.english_out.write(
                    str((-1 if with_minus else 1)*rad_angles[k]*180/np.pi) +
                    "\n" if k == (num_angles-1) else "\t")
        
        # picture file
        pic_line = ""
        biggest = aft_tar_bit_pos
        smallest = aft_tar_bit_pos
        if num_controls != 0:
            biggest = max(aft_trols.bit_pos[0], aft_tar_bit_pos)
            smallest = min(aft_trols.bit_pos[num_controls-1], aft_tar_bit_pos)

        # k a bit position
        for k in range(num_bits_aft-1, biggest, -1):
            pic_line += "|   "

        c_int = 0
        for k in range(biggest, smallest-1, -1):
            is_target = (k == tar_bit_pos)
            is_control = False
            control_kind = False
            tres = ":" if (k == smallest) else "---"
            dos = ":" if (k == smallest) else "--"

            # c_int starts at last value
            for c in range(c_int, num_controls, +1):
                if k == aft_trols.bit_pos[c]:
                    is_control = True
                    control_kind = aft_trols.kinds[c]
                    c_int += 1
                    break

            if is_control:
                if isinstance(control_kind, int):
                    pic_line += "@O" + dos
                elif not control_kind:
                    pic_line += "O" + tres
                elif control_kind:
                    pic_line += "@" + tres
            else:  # is not control
                if not is_target:  # is not control nor target
                    pic_line += "+" + tres
                else:  # is target
                    pic_line += "Ry" + dos
        
        for k in range(smallest-1, -1, -1):
            pic_line += "|   "
            
        self.write_pic_line(pic_line)
        self.picture_out.write("\n")
        
    def write_bit_swap(self, bit1, bit2):
        """
        Write a line in eng & pic files for a 'SWAP' with no controls.

        Parameters
        ----------
        bit1 : int
        bit2 : int

        Returns
        -------
        None

        """
        trols = Controls(self.num_bits) # dummy with zero controls
        self.write_controlled_bit_swap(bit1, bit2, trols)

    def write_one_bit_gate(
            self, tar_bit_pos, one_bit_gate_fun, fun_arg_list):
        """
        Write a line in eng & pic files for a one qubit gate (from class
        OneBitGates) with no controls.

        Parameters
        ----------
        tar_bit_pos : int
        one_bit_gate_fun : function
        fun_arg_list : list[]

        Returns
        -------
        None

        """
        trols = Controls(self.num_bits)  # dummy with zero controls
        self.write_controlled_one_bit_gate(
            tar_bit_pos, trols, one_bit_gate_fun, fun_arg_list)
        
    def write_global_phase_fac(self, ang_rads):
        """
        Write a line in eng & pic files for a global phase factor 'PHAS'
        with no controls.

        Parameters
        ----------
        ang_rads : float

        Returns
        -------
        None

        """
        tar_bit_pos = 0  # anyone will do
        trols = Controls(self.num_bits)  # dummy with zero controls
        gate_fun = OneBitGates.one_bit_phase_fac
        self.write_controlled_one_bit_gate(
            tar_bit_pos, trols, gate_fun, [ang_rads])
        
    def write_mulitplexor_gate(
            self, tar_bit_pos, controls, with_minus, rad_angles):
        """
        Write a line in eng & pic files for a multiplexor 'MP_Y' with
        no T/F controls.

        Parameters
        ----------
        tar_bit_pos : int
        controls : Controls
        with_minus : bool
        rad_angles : list[float]

        Returns
        -------
        None

        """
        num_controls = len(controls.bit_pos)
        assert num_controls == controls.get_num_int_controls(),\
            "some of the controls of this multiplexor are not half-moons"
        self.write_controlled_multiplexor_gate(
            tar_bit_pos, controls, with_minus, rad_angles)

if __name__ == "__main__":
    print(5)

