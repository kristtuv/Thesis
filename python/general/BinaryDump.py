import struct
import numpy as np
import sys

class BinaryDump():
    def __init__(self, filename, headernames=False):
        self.filename = filename
        self.headernames = headernames
        self.file_object = open(filename, 'r')
        self._indexFile()

    def _indexFile(self):
        self.filePos = []
        self.timesteps = []
        while not self.getNextTimestep(indexing=True)[0] == None:
            pass
            # print("reading timestep")
        self.filePos = np.asarray(self.filePos)
        self.timesteps = np.asarray(self.timesteps)
        self.file_object.seek(0)

    def seekToTimestep(self, frame):
        self.file_object.seek(self.filePos[frame])

    def getNextTimestep(self, indexing=False):
        header, atomdata =  self.readTimestep(indexing)
        if atomdata is None:
            return None, None
        return header, atomdata

    def readTimestep(self, indexing = False):
        inposition = self.file_object.tell()

        myHeaderType = np.dtype([("timestep", np.int64),
                            ("nAtoms", np.int64),
                            ("triclinic", np.int32),
                            ("boundary1", np.int32),
                            ("boundary2", np.int32),
                            ("boundary3", np.int32),
                            ("boundary4", np.int32),
                            ("boundary5", np.int32),
                            ("boundary6", np.int32),
                            ("xlo", np.float64),
                            ("xhi", np.float64),
                            ("ylo", np.float64),
                            ("yhi", np.float64),
                            ("zlo", np.float64),
                            ("zhi", np.float64),
                            ("sizeOne", np.int32),
                            ("nChunk", np.int32)])
        header = np.fromfile(self.file_object, dtype=myHeaderType, count=1)

        if not header:
            return None, None
        if indexing:
            self.filePos.append(inposition)
            self.timesteps.append(header["timestep"])

        if header["triclinic"][0]:
            self.file_object.seek(inposition)
            myHeaderType = np.dtype([("timestep", np.int64),
                                ("nAtoms", np.int64),
                                ("triclinic", np.int32),
                                ("boundary1", np.int32),
                                ("boundary2", np.int32),
                                ("boundary3", np.int32),
                                ("boundary4", np.int32),
                                ("boundary5", np.int32),
                                ("boundary6", np.int32),
                                ("xlo", np.float64),
                                ("xhi", np.float64),
                                ("ylo", np.float64),
                                ("yhi", np.float64),
                                ("zlo", np.float64),
                                ("zhi", np.float64),
                                ("xy", np.float64),
                                ("xz", np.float64),
                                ("yz", np.float64),
                                ("sizeOne", np.int32),
                                ("nChunk", np.int32)])

            header = np.fromfile(self.file_object, dtype=myHeaderType, count=1)
        header_dict = {}
        for key in myHeaderType.fields.keys():
            header_dict[key] = header[key][0]

        atomdata = np.asarray([])
        for i in range(header_dict["nChunk"]):
            nEntries = np.fromfile(self.file_object, np.int32, 1)[0]
            atomdata = np.append(atomdata,np.fromfile(self.file_object, dtype=np.float64, count=nEntries))
        atomdata = atomdata.reshape((header_dict["nAtoms"], header_dict["sizeOne"]))
        shape = np.shape(atomdata)
        return header_dict, atomdata
