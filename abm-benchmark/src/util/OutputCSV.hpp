#pragma once 

#include <cstdio>
#include <string>
#include <vector>

namespace util {
class OutputCSV { 
 public:
    OutputCSV(FILE * fp) : 
        header({}),
        rows({}),
        fp(fp)
        {
    };
    ~OutputCSV() { };

    void setHeader(std::string header) {
        this->header = header;
    };
    void appendRow(std::string row) {
        this->rows.push_back(row);
    };
    void writeHeader() {
        if (this->header.size()) {
            FILE * fp = this->fp != nullptr ? this->fp : stdout;
            fprintf(fp, "%s\n", this->header.c_str());
            fflush(fp);
        }
    };
    void writeRows() {
        FILE * fp = this->fp != nullptr ? this->fp : stdout;
        for (const auto & row : this->rows) {
            fprintf(fp, "%s\n", row.c_str());
        }
        fflush(fp);
    };
    void write() {
        this->writeHeader();
        this->writeRows();
    };
    void clear() {
        this->rows.clear(); 
    };
 private:
    std::string header;
    std::vector<std::string> rows;
    FILE * fp;
};
}  // namespace util