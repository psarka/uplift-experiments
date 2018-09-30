import json


class JsonWriter:

    def __init__(self,
                 file_handle):

        self.file_handle = file_handle

    def writerow(self, record):

        self.file_handle.write(json.dumps(record))
        self.file_handle.write('\n')

    def writerows(self, records):

        for r in records:
            self.writerow(r)


class JsonReader:

    def __init__(self,
                 file_handle):

        self.file_handle = file_handle
        self.rows = iter(file_handle)

    def __iter__(self):
        return self

    def __next__(self):
        return json.loads(next(self.rows))

