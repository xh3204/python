import xlrd
import xlwt

def read_excel_all(path, sheet):
    with xlrd.open_workbook(path) as f:
        table = f.sheet_by_name(str(sheet))
    data = []
    for idx in range(table.nrows):
        row = table.row_values(idx)
        if row:
            data.append(row)
    return data

# read the specified row and column data of excel
def read_excel_row_column(path, sheet, row_start=0, row_stop=0, column_start=0, column_stop=0):
    with xlrd.open_workbook(path) as f:
        table = f.sheet_by_name(str(sheet))
    data = []
    for idx in range(row_start-1, row_stop):
        row = table.row_values(idx)
        column = []
        if row:
            if row_stop > len(row):
                for row_idx in range(column_start-1, column_stop):
                    column.append(row[row_idx])
                data.append(column)
            else:
                print('column number is invalid')
    return data

if __name__ == '__main__':
    data = read_excel_row_column(r'D:\学习\博士\谣言传播仿真\文档\hep\hep_directed.xlsx', 'Sheet1',62,65,3,6)
    print(data)
