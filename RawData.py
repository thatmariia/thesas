from GraphParser import NodeType, EdgeDistType, EdgeConnectionType, Node, Edge
import numpy as np


class RawData:

    def __init__(self, scale=1):
        self.scale = scale

        self.max_width = 40
        self.max_height = 40

        # virtual nodes (for hyper-edges)
        VOUT = Node("VOUT", NodeType.VIRTUAL)
        VIN = Node("VIN", NodeType.VIRTUAL)
        SW = Node("SW", NodeType.VIRTUAL)
        FB = Node("FB", NodeType.VIRTUAL)
        GND = Node("GND", NodeType.VIRTUAL)
        VS = Node("VS", NodeType.VIRTUAL)

        # assumption for side lengths: the side lengths are divisible by the nr of slots on the side
        # component CO
        COV = Node("COV", NodeType.REAL)
        COG = Node("COG", NodeType.REAL)
        # component CO borders
        CO_1 = Node("CO_1", NodeType.BORDER)
        CO_2 = Node("CO_2", NodeType.BORDER)
        CO_3 = Node("CO_3", NodeType.BORDER)
        CO_4 = Node("CO_4", NodeType.BORDER)
        # component CO side lengths
        CO_w = 4 * scale
        CO_h = 5 * scale

        # component JO
        JOV = Node("JOV", NodeType.REAL)
        JOG = Node("JOG", NodeType.REAL)
        # component JO borders
        JO_1 = Node("JO_1", NodeType.BORDER)
        JO_2 = Node("JO_2", NodeType.BORDER)
        JO_3 = Node("JO_3", NodeType.BORDER)
        JO_4 = Node("JO_4", NodeType.BORDER)
        # component JO side lengths
        JO_w = 10 * scale
        JO_h = 17 * scale

        # component L1
        L1L = Node("L1L", NodeType.REAL)
        L1H = Node("L1H", NodeType.REAL)
        # component L1 borders
        L1_1 = Node("L1_1", NodeType.BORDER)
        L1_2 = Node("L1_2", NodeType.BORDER)
        L1_3 = Node("L1_3", NodeType.BORDER)
        L1_4 = Node("L1_4", NodeType.BORDER)
        # component L1 side lengths
        L1_w = 14 * scale
        L1_h = 12 * scale

        # component JI
        JIG = Node("JIG", NodeType.REAL)
        JIV = Node("JIV", NodeType.REAL)
        # component JI borders
        JI_1 = Node("JI_1", NodeType.BORDER)
        JI_2 = Node("JI_2", NodeType.BORDER)
        JI_3 = Node("JI_3", NodeType.BORDER)
        JI_4 = Node("JI_4", NodeType.BORDER)
        # component JI side lengths
        JI_w = 10 * scale
        JI_h = 17 * scale

        # component CI
        CIG = Node("CIG", NodeType.REAL)
        CIV = Node("CIV", NodeType.REAL)
        # component CI borders
        CI_1 = Node("CI_1", NodeType.BORDER)
        CI_2 = Node("CI_2", NodeType.BORDER)
        CI_3 = Node("CI_3", NodeType.BORDER)
        CI_4 = Node("CI_4", NodeType.BORDER)
        # component CI side lengths
        CI_w = 4 * scale
        CI_h = 5 * scale

        # component U
        UVI = Node("UVI", NodeType.REAL)
        USW = Node("USW", NodeType.REAL)
        UG = Node("UG", NodeType.REAL)
        UFB = Node("UFB", NodeType.REAL)
        UEN = Node("UEN", NodeType.REAL)
        UVS = Node("UVS", NodeType.REAL)
        # component U borders
        U_1 = Node("U_1", NodeType.BORDER)
        U_2 = Node("U_2", NodeType.BORDER)
        U_3 = Node("U_3", NodeType.BORDER)
        U_4 = Node("U_4", NodeType.BORDER)
        # component U side lengths
        U_w = 4 * scale
        U_h = 4 * scale

        # component C1
        CSW = Node("CSW", NodeType.REAL)
        CVS = Node("CVS", NodeType.REAL)
        # component C1 borders
        C1_1 = Node("C1_1", NodeType.BORDER)
        C1_2 = Node("C1_2", NodeType.BORDER)
        C1_3 = Node("C1_3", NodeType.BORDER)
        C1_4 = Node("C1_4", NodeType.BORDER)
        # component C1 side lengths
        C1_w = 5 * scale
        C1_h = 4 * scale

        # component R1
        R1V = Node("R1V", NodeType.REAL)
        R1FB = Node("R1FB", NodeType.REAL)
        # component R1 borders
        R1_1 = Node("R1_1", NodeType.BORDER)
        R1_2 = Node("R1_2", NodeType.BORDER)
        R1_3 = Node("R1_3", NodeType.BORDER)
        R1_4 = Node("R1_4", NodeType.BORDER)
        # component R1 side lengths
        R1_w = 5 * scale
        R1_h = 4 * scale

        # component R2
        R2FB = Node("R2FB", NodeType.REAL)
        R2G = Node("R2G", NodeType.REAL)
        # component R2 borders
        R2_1 = Node("R2_1", NodeType.BORDER)
        R2_2 = Node("R2_2", NodeType.BORDER)
        R2_3 = Node("R2_3", NodeType.BORDER)
        R2_4 = Node("R2_4", NodeType.BORDER)
        # component R2 side lengths
        R2_w = 6 * scale
        R2_h = 4 * scale

        self.vertices = [
            CO_1, CO_2, CO_3, CO_4, COV, COG,
            JO_1, JO_2, JO_3, JO_4, JOG, JOV,
            L1_1, L1_2, L1_3, L1_4, L1L, L1H,
            JI_1, JI_2, JI_3, JI_4, JIG, JIV,
            CI_1, CI_2, CI_3, CI_4, CIG, CIV,
            U_1, U_2, U_3, U_4, UVI, USW, UG, UFB, UEN, UVS,
            C1_1, C1_2, C1_3, C1_4, CSW, CVS,
            R1_1, R1_2, R1_3, R1_4, R1V, R1FB,
            R2_1, R2_2, R2_3, R2_4, R2FB, R2G,
            VOUT, VIN, SW, FB, GND, VS
        ]

        diag_tolerane = 1

        def diag(side1_len, side2_len, nr_elements_side1=1, nr_elements_side2=1):
            d_side1_len = int(round(side1_len / nr_elements_side1))
            new_side1_len = d_side1_len * nr_elements_side1
            d_side2_len = int(round(side2_len / nr_elements_side2))
            new_side2_len = d_side2_len * nr_elements_side2
            return int(np.sqrt(new_side1_len * 2 + new_side2_len * 2))

        self.component_bounds = [
            # order: top left, top right, bottom left, bottom right
            (CO_1.name, CO_2.name, CO_3.name, CO_4.name),
            (JO_1.name, JO_2.name, JO_3.name, JO_4.name),
            (L1_1.name, L1_2.name, L1_3.name, L1_4.name),
            (JI_1.name, JI_2.name, JI_3.name, JI_4.name),
            (CI_1.name, CI_2.name, CI_3.name, CI_4.name),
            (U_1.name, U_2.name, U_3.name, U_4.name),
            (C1_1.name, C1_2.name, C1_3.name, C1_4.name),
            (R1_1.name, R1_2.name, R1_3.name, R1_4.name),
            (R2_1.name, R2_2.name, R2_3.name, R2_4.name)
        ]

        self.component_edges = [
            [
                # ------------------------------- component CO
                # horizontal edges
                Edge((CO_1, COV), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CO_w / 2),
                Edge((COV, CO_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CO_w / 2),
                Edge((CO_3, COG), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CO_w / 2),
                Edge((COG, CO_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CO_w / 2),
                Edge((CO_1, CO_2), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, CO_w),
                Edge((CO_3, CO_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, CO_w),
                # vertical edges
                Edge((CO_1, CO_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CO_h),
                Edge((CO_2, CO_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CO_h),
                # diagonal edges
                Edge((CO_1, CO_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(CO_w, CO_h, 2, 1) - diag_tolerane),
                Edge((CO_4, CO_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(CO_w, CO_h, 2, 1) + diag_tolerane),
                Edge((CO_2, CO_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(CO_w, CO_h, 2, 1) - diag_tolerane),
                Edge((CO_3, CO_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(CO_w, CO_h, 2, 1) + diag_tolerane),
                # interport edges
                Edge((COV, COG), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, CO_h)
            ],
            [
                # ------------------------------- component JO
                # horizontal edges
                Edge((JO_1, JOV), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JO_w / 2),
                Edge((JOV, JO_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JO_w / 2),
                Edge((JO_3, JOG), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JO_w / 2),
                Edge((JOG, JO_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JO_w / 2),
                Edge((JO_1, JO_2), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, JO_w),
                Edge((JO_3, JO_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, JO_w),
                # vertical edges
                Edge((JO_1, JO_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JO_h),
                Edge((JO_2, JO_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JO_h),
                # diagonal edges
                Edge((JO_1, JO_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(JO_w, JO_h, 2, 1) - diag_tolerane),
                Edge((JO_4, JO_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(JO_w, JO_h, 2, 1) + diag_tolerane),
                Edge((JO_2, JO_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(JO_w, JO_h, 2, 1) - diag_tolerane),
                Edge((JO_3, JO_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(JO_w, JO_h, 2, 1) + diag_tolerane),
                # interport edges
                Edge((JOV, JOG), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, JO_h)
            ],
            [
                # ------------------------------- component L1
                # horizontal edges
                Edge((L1_1, L1_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, L1_w),
                Edge((L1_3, L1_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, L1_w),
                # vertical edges
                Edge((L1_1, L1L), EdgeDistType.FIXED, EdgeConnectionType.BORDER, L1_h / 2),
                Edge((L1L, L1_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, L1_h / 2),
                Edge((L1_2, L1H), EdgeDistType.FIXED, EdgeConnectionType.BORDER, L1_h / 2),
                Edge((L1H, L1_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, L1_h / 2),
                Edge((L1_1, L1_3), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, L1_h),
                Edge((L1_2, L1_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, L1_h),
                # diagonal edges
                Edge((L1_1, L1_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(L1_w, L1_h, 1, 2) - diag_tolerane),
                Edge((L1_4, L1_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(L1_w, L1_h, 1, 2) + diag_tolerane),
                Edge((L1_2, L1_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(L1_w, L1_h, 1, 2) - diag_tolerane),
                Edge((L1_3, L1_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(L1_w, L1_h, 1, 2) + diag_tolerane),
                # interport edges
                Edge((L1L, L1H), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, L1_w)
            ],
            [
                # ------------------------------- component JI
                # horizontal edges
                Edge((JI_1, JIV), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JI_w / 2),
                Edge((JIV, JI_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JI_w / 2),
                Edge((JI_3, JIG), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JI_w / 2),
                Edge((JIG, JI_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JI_w / 2),
                Edge((JI_1, JI_2), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, JI_w),
                Edge((JI_3, JI_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, JI_w),
                # vertical edges
                Edge((JI_1, JI_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JI_h),
                Edge((JI_2, JI_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, JI_h),
                # diagonal edges
                Edge((JI_1, JI_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(JI_w, JI_h, 2, 1) - diag_tolerane),
                Edge((JI_4, JI_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(JI_w, JI_h, 2, 1) + diag_tolerane),
                Edge((JI_2, JI_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(JI_w, JI_h, 2, 1) - diag_tolerane),
                Edge((JI_3, JI_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(JI_w, JI_h, 2, 1) + diag_tolerane),
                # interport edges
                Edge((JIV, JIG), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, JI_h)
            ],
            [
                # ------------------------------- component CI
                # horizontal edges
                Edge((CI_1, CIV), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CI_w / 2),
                Edge((CIV, CI_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CI_w / 2),
                Edge((CI_3, CIG), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CI_w / 2),
                Edge((CIG, CI_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CI_w / 2),
                Edge((CI_1, CI_2), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, CI_w),
                Edge((CI_3, CI_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, CI_w),
                # vertical edges
                Edge((CI_1, CI_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CI_h),
                Edge((CI_2, CI_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, CI_h),
                # diagonal edges
                Edge((CI_1, CI_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(CI_w, CI_h, 2, 1) - diag_tolerane),
                Edge((CI_4, CI_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(CI_w, CI_h, 2, 1) + diag_tolerane),
                Edge((CI_2, CI_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(CI_w, CI_h, 2, 1) - diag_tolerane),
                Edge((CI_3, CI_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(CI_w, CI_h, 2, 1) + diag_tolerane),
                # interport edges
                Edge((CIV, CIG), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, CI_h)
            ],
            [
                # ------------------------------- component U
                # horizontal edges
                Edge((U_1, U_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_w),
                Edge((U_3, U_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_w),
                # vertical edges
                Edge((U_1, UG), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((UG, USW), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((USW, UVI), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((UVI, U_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((U_2, UVS), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((UVS, UEN), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((UEN, UFB), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((UFB, U_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, U_h / 4),
                Edge((U_1, U_3), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, U_h),
                Edge((U_2, U_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, U_h),
                # diagonal edges
                Edge((U_1, U_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(U_w, U_h, 1, 4) - diag_tolerane),
                Edge((U_4, U_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(U_w, U_h, 1, 4) + diag_tolerane),
                Edge((U_2, U_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(U_w, U_h, 1, 4) - diag_tolerane),
                Edge((U_3, U_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(U_w, U_h, 1, 4) + diag_tolerane),
                # interport edges
                Edge((UG, UVS), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, U_w),
                Edge((USW, UEN), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, U_w),
                Edge((UVI, UFB), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, U_w)
            ],
            [
                # ------------------------------- component C1
                # horizontal edges
                Edge((C1_1, C1_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, C1_w),
                Edge((C1_3, C1_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, C1_w),
                # vertical edges
                Edge((C1_1, CSW), EdgeDistType.FIXED, EdgeConnectionType.BORDER, C1_h / 2),
                Edge((CSW, C1_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, C1_h / 2),
                Edge((C1_2, CVS), EdgeDistType.FIXED, EdgeConnectionType.BORDER, C1_h / 2),
                Edge((CVS, C1_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, C1_h / 2),
                Edge((C1_1, C1_3), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, C1_h),
                Edge((C1_2, C1_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, C1_h),
                # diagonal edges
                Edge((C1_1, C1_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(C1_w, C1_h, 1, 2) - diag_tolerane),
                Edge((C1_4, C1_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(C1_w, C1_h, 1, 2) + diag_tolerane),
                Edge((C1_2, C1_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(C1_w, C1_h, 1, 2) - diag_tolerane),
                Edge((C1_3, C1_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(C1_w, C1_h, 1, 2) + diag_tolerane),
                # interport edges
                Edge((CSW, CVS), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, C1_w)
            ],
            [
                # ------------------------------- component R1
                # horizontal edges
                Edge((R1_1, R1_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R1_w),
                Edge((R1_3, R1_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R1_w),
                # vertical edges
                Edge((R1_1, R1V), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R1_h / 2),
                Edge((R1V, R1_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R1_h / 2),
                Edge((R1_2, R1FB), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R1_h / 2),
                Edge((R1FB, R1_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R1_h / 2),
                Edge((R1_1, R1_3), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, R1_h),
                Edge((R1_2, R1_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, R1_h),
                # diagonal edges
                Edge((R1_1, R1_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(R1_w, R1_h, 1, 2) - diag_tolerane),
                Edge((R1_4, R1_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(R1_w, R1_h, 1, 2) + diag_tolerane),
                Edge((R1_2, R1_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(R1_w, R1_h, 1, 2) - diag_tolerane),
                Edge((R1_3, R1_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(R1_w, R1_h, 1, 2) + diag_tolerane),
                # interport edges
                Edge((R1V, R1FB), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, R1_w)
            ],
            [
                # ------------------------------- component R2
                # horizontal edges
                Edge((R2_1, R2FB), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R2_w / 2),
                Edge((R2FB, R2_2), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R2_w / 2),
                Edge((R2_3, R2G), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R2_w / 2),
                Edge((R2G, R2_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R2_w / 2),
                Edge((R2_1, R2_2), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, R2_w),
                Edge((R2_3, R2_4), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, R2_w),
                # vertical edges
                Edge((R2_1, R2_3), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R2_h),
                Edge((R2_2, R2_4), EdgeDistType.FIXED, EdgeConnectionType.BORDER, R2_h),
                # diagonal edges
                Edge((R2_1, R2_4), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(R2_w, R2_h, 2, 1) - diag_tolerane),
                Edge((R2_4, R2_1), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(R2_w, R2_h, 2, 1) + diag_tolerane),
                Edge((R2_2, R2_3), EdgeDistType.MORE_THAN, EdgeConnectionType.DISTANCE,
                     diag(R2_w, R2_h, 2, 1) - diag_tolerane),
                Edge((R2_3, R2_2), EdgeDistType.LESS_THAN, EdgeConnectionType.DISTANCE,
                     diag(R2_w, R2_h, 2, 1) + diag_tolerane),
                # interport edges
                Edge((R2FB, R2G), EdgeDistType.FIXED, EdgeConnectionType.DISTANCE, R2_h)
            ]
        ]

        flattened_component_edges = [item for sublist in self.component_edges for item in sublist]
        self.edges = flattened_component_edges + [
            # wire edges
            Edge((VS, CVS), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((VS, UVS), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),

            Edge((VOUT, COV), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((VOUT, L1L), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((VOUT, R1V), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((VOUT, JOV), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),

            Edge((VIN, UVI), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((VIN, JIV), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((VIN, CIV), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),

            Edge((SW, L1H), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((SW, CSW), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((SW, USW), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),

            Edge((FB, R1FB), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((FB, R2FB), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((FB, UFB), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),

            Edge((GND, UG), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((GND, JOG), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((GND, CIG), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((GND, JIG), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((GND, COG), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),
            Edge((GND, R2G), EdgeDistType.UNIMPORTANT, EdgeConnectionType.WIRE),

            # distance edges
            Edge((CIV, UVI), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((COV, USW), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R1V, USW), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R1V, CSW), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R1V, L1H), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R2G, USW), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R2G, CSW), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R2G, L1H), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((CVS, UVS), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            # Edge((UFB, L1H), EdgeDistType.MAXIMIZE, EdgeConnectionType.DISTANCE),
            Edge((L1H, USW), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((L1H, CSW), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R1FB, R2FB), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R1FB, UFB), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R2FB, UFB), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE),
            Edge((R1FB, R2FB), EdgeDistType.MINIMIZE, EdgeConnectionType.DISTANCE)
        ]
