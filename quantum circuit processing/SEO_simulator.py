import numpy as np
from QLib.src.SEO_reader import *
from QLib.src.OneBitGates import *


class SEO_simulator(SEO_reader):
    """
    This class simulates the evolution of a quantum state vector. The
    initial state vector can be inputted to the constructor or it can be set
    to the ground state by a function provided for doing this.

    This class has SEO_reader as a parent. Each line of an English file is 
    read by the parent class and handed over to the use_ functions of this 
    class. The use functions multiply the current state vector by the 
    unitary matrix that represents the latest line read. 

    Attributes
    ----------
    cur_st_vec : np.ndarray
        current state vector

    num_ops : int
    loop2cur_rep : dict[int, int]

    english_in : _io.TextIOWrapper
    file_prefix : str
    loop2start_offset : dict[int, int]
    loop2tot_reps : dict[int, int]
    loop_queue : list[int]
    num_bits : int
    num_lines : int
    split_line : list[str]

    """

    # combines java classes:
    # LineList, UnitaryMat, SEO_readerMu

    def __init__(self, file_prefix, num_bits, init_st_vec=None):
        """
        Constructor

        Parameters
        ----------
        file_prefix : str
        num_bits : int

        Returns
        -------

        """

        SEO_reader.__init__(self, file_prefix, num_bits)
        self.cur_st_vec = init_st_vec

    def set_cur_st_vec_to_ground_st(self):
        """
        Sets current state vector to ground state |0>|0>|0>...|0>, where |0>
        = [1,0]^t and |1> = [0,1]^t, t = transpose

        Returns
        -------
        None

        """
        ty = np.complex128
        mat = np.zeros([1 << self.num_bits], dtype=ty)
        mat[tuple([0]*self.num_bits)] = 1
        mat.reshape([2]*self.num_bits)
        self.cur_st_vec = mat

    def evolve_by_controlled_bit_swap(self, bit1, bit2, controls):
        """
        Evolve current state vector by controlled bit swap.

        Parameters
        ----------
        bit1 : int
        bit2 : int
        controls : Controls

        Returns
        -------

        """
        assert bit1 != bit2, "swapped bits must be different"
        for bit in [bit1, bit2]:
            assert -1 < bit < self.num_bits
            assert bit not in controls.bit_pos

        slicex = [slice(None)]*self.num_bits
        num_controls = len(controls.bit2kind)
        for k in range(num_controls):
            assert isinstance(controls.bit_pos[k], bool)
            if controls.kinds[k]:  # it's True
                slicex[controls.bit_pos[k]] = 1
            else:  # it's False
                slicex[controls.bit_pos[k]] = 0
        slicex = tuple(slicex)
        vec = self.cur_st_vec[slicex]
        self.cur_st_vec[slicex] = np.transpose(vec, (bit1, bit2))

    def evolve_by_controlled_one_bit_gate(self,
                tar_bit_pos, controls, one_bit_gate):
        """
        Evolve current state vector by controlled one bit gate (from class
        OneBitGates). Note one_bit_gate inputted as np.array.

        Parameters
        ----------
        tar_bit_pos : int
        controls : Controls
        one_bit_gate : np.ndarray

        Returns
        -------

        """
        assert tar_bit_pos not in controls.bit_pos
        assert -1 < tar_bit_pos < self.num_bits

        slicex = [slice(None)]*self.num_bits
        num_controls = len(controls.bit2kind)
        for k in range(num_controls):
            assert isinstance(controls.bit_pos[k], bool)
            if controls.kinds[k]:  # it's True
                slicex[controls.bit_pos[k]] = 1
            else:  # it's False
                slicex[controls.bit_pos[k]] = 0
        slicex = tuple(slicex)
        vec = self.cur_st_vec[slicex]
        shape = [1]*tar_bit_pos + [2] + \
                [1]*(self.num_bits-tar_bit_pos-1) + [2]
        mat = one_bit_gate.reshape(shape)
        self.cur_st_vec[slicex] = np.dot(mat, vec)
        
    def use_NOTA(self, bla_str):
        """
        Overrides the parent class use_ function. Does nothing.

        Parameters
        ----------
        bla_str : str

        Returns
        -------

        """
        pass

    # def use_LOOP(self, loop_num, reps):

    # def use_NEXT(self, loop_num):

    def use_SWAP(self, bit1, bit2, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_bit_swap().

        Parameters
        ----------
        bit1 : int
        bit2 : int
        controls : Controls

        Returns
        -------
        None

        """
        self.evolve_by_controlled_bit_swap(bit1, bit2, controls)

    def use_PHAS(self, angle_degs, tar_bit_pos, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_one_bit_gate()
        for PHAS.

        Parameters
        ----------
        angle_degs : float
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        gate = OneBitGates.one_bit_phase_fac(angle_degs*np.pi/180)
        self.evolve_by_controlled_one_bit_gate(tar_bit_pos, controls, gate)

    def use_P_PH(self, projection_bit,
                angle_degs, tar_bit_pos, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_one_bit_gate()
        for P_0 and P_1 phase factors.


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
        g = {
                0: OneBitGates.one_bit_P_0_phase_fac,
                1: OneBitGates.one_bit_P_1_phase_fac
            }
        gate = g[projection_bit](angle_degs*np.pi/180)
        self.evolve_by_controlled_one_bit_gate(tar_bit_pos, controls, gate)

    def use_SIG(self, direction, tar_bit_pos, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_one_bit_gate()
        for sigx, sigy, sigz.

        Parameters
        ----------
        direction : int
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        s = {
                1: OneBitGates.sigx,
                2: OneBitGates.sigy,
                3: OneBitGates.sigz
            }
        gate = s[direction]()
        self.evolve_by_controlled_one_bit_gate(tar_bit_pos, controls, gate)

    def use_HAD2(self, tar_bit_pos, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_one_bit_gate()
        for had2.


        Parameters
        ----------
        tar_bit_pos : int
        controls : Controls

        Returns
        -------
        None

        """
        gate = OneBitGates.had2()
        self.evolve_by_controlled_one_bit_gate(tar_bit_pos, controls, gate)

    def use_ROT(self, direction,
                angle_degs, tar_bit_pos, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_one_bit_gate()
        for rot along axes x, y, or z.


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
        gate = OneBitGates.one_bit_rot_ax(angle_degs*np.pi/180, direction)
        self.evolve_by_controlled_one_bit_gate(tar_bit_pos, controls, gate)

    def use_ROTN(self, angle_x_degs, angle_y_degs, angle_z_degs,
                tar_bit_pos, controls):
        """
        Overrides the parent class use_ function. Calls evolve_by_controlled_one_bit_gate()
        for rot along arbitrary direction.


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
        gate = OneBitGates.one_bit_rot(
                    angle_x_degs*np.pi/180,
                    angle_y_degs*np.pi/180,
                    angle_z_degs*np.pi/180)
        self.evolve_by_controlled_one_bit_gate(tar_bit_pos, controls, gate)

if __name__ == "__main__":
    print(5)