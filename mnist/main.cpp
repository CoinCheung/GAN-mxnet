#include<fstream>
#include<string>
#include<sstream>
#include<iostream>
#include<vector>
#include<cassert>
#include<opencv2/opencv.hpp>


std::vector<std::string> extract_mnist_imgs(const char* fn_c, const char* pth_c);
std::vector<int> extract_mnist_lbs(const char* fn_c);
void dump_txt(std::vector<std::string>&, std::vector<int>&, const char*);
int get_file_len(std::ifstream&);
int ReverseInt(int);


int main() {
    using namespace std;

    cout << "processing training set\n";
    auto train_im = extract_mnist_imgs("../train-images-idx3-ubyte", "../images/train/");
    auto train_lb = extract_mnist_lbs("../train-labels-idx1-ubyte");
    dump_txt(train_im, train_lb, "../images/train.txt");

    cout << "processing val set\n";
    auto val_im = extract_mnist_imgs("../t10k-images-idx3-ubyte", "../images/val/");
    auto val_lb = extract_mnist_lbs("../t10k-labels-idx1-ubyte");
    dump_txt(val_im, val_lb, "../images/val.txt");

    return 0;
}


std::vector<std::string> extract_mnist_imgs(const char* fn, const char* pth_c) {
    using namespace std;

    int len;
    int num;
    int n_row, n_col;
    string pth(pth_c);

    ifstream fin;
    fin.open(fn, ios_base::in);
    if (!fin.is_open()) {
        cout << "open failed !!\n";
        assert(false);
    }

    // read data information
    len = get_file_len(fin);
    fin.seekg(4); // skip magic number
    fin.read(reinterpret_cast<char*>(&num), 4);
    num = ReverseInt(num);
    fin.read(reinterpret_cast<char*>(&n_row), 4);
    n_row = ReverseInt(n_row);
    fin.read(reinterpret_cast<char*>(&n_col), 4);
    n_col = ReverseInt(n_col);
    cout << "length: " << len << ", image num: " << num << endl;
    cout << "image number of rows: " << n_row << ", image number of rows: " << n_col << endl;

    // read images
    int img_size = n_row * n_col;
    int buf_size = img_size * num;
    int off_st;
    vector<char> buffer(buf_size);
    vector<char> one(img_size);
    auto size = cv::Size(n_row, n_col);
    stringstream ss;
    string iname;
    vector<string> im_names;
    
    fin.read(&buffer[0], buf_size);
    auto st = buffer.begin();
    for (auto i{0}; i < num; ++i) {
        off_st = i * img_size;
        one.assign(st + off_st, st + off_st + img_size);

        ss.clear();
        ss.str("");
        ss << "train_" << i << ".jpg";
        im_names.emplace_back(ss.str());
        cv::imwrite(pth + ss.str(), cv::Mat(size, CV_8UC1, &one[0]));
    }

    fin.close();
    return im_names;
}


std::vector<int> extract_mnist_lbs(const char* fn_c) {
    using namespace std;

    string fn(fn_c);
    ifstream fin;
    int num;
    char ch;

    fin.open(fn, ios_base::in);
    if (!fin.is_open()) {
        cout << "file open failed !\n";
        assert(false);
    }

    fin.seekg(4);
    fin.read(reinterpret_cast<char*>(&num), 4);
    num = ReverseInt(num);
    cout << "label numbers: " << num << endl;

    vector<int> labels(num);
    for (int i{0}; i < num; ++i) {
        fin.read(&ch, 1);
        labels[i] = static_cast<int>(ch);
    }

    fin.close();
    return labels;
}


void dump_txt(std::vector<std::string>& iname, std::vector<int>& labels, const char* pth) {
    using namespace std;

    int n_name, n_lb;
    ofstream fout;
    string fn(pth);

    n_name = static_cast<int>(iname.size());
    n_lb = static_cast<int>(labels.size());
    if (n_lb != n_name) {
        cout << "number of labels and images not match !!" << endl;
        assert(false);
    }
    fout.open(fn, ios_base::out);
    if (!fout.is_open()) {
        cout << "file open failed !!" << endl;
        assert(false);
    }

    for (int i{0}; i < n_name; ++i) {
        fout << labels[i] << ", " << iname[i] << endl;
    }
    fout.close();
}


int get_file_len(std::ifstream& fin) {
    using namespace std;
    int len;
    int curr;

    curr = fin.tellg();
    fin.seekg(0, fin.end);
    len = fin.tellg();
    fin.clear();
    fin.seekg(curr);
    
    return len;
}


int ReverseInt(int n) {
    int b1, b2, b3, b4;
    b1 = static_cast<int>(n & 0xff);
    b2 = static_cast<int>(n >> 8) & 0xff;
    b3 = static_cast<int>(n >> 16) & 0xff;
    b4 = static_cast<int>(n >> 24) & 0xff;

    return (b1 << 24) + (b2 << 16) + (b3 << 8) + b4;
}
