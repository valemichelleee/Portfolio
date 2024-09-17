class DataIO
{
public:
    // Constructor
    DataIO();

    // Destructor
    ~DataIO();

    // read metadata for images or labels
    Eigen::MatrixXd readTensor(std::string input_file);

    // write tensor in file
    void writeTensorToFile(const Eigen::MatrixXd &tensor, std::string output_file, int required_element_index);

private:
    // swap endiannes (big-endian to little-endian)
    inline uint32_t swapEndian(uint32_t value);
};

// Constructor
DataIO::DataIO()
{
}

// Destructor
DataIO::~DataIO()
{
}

uint32_t DataIO::swapEndian(uint32_t value)
{
    return ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | ((value >> 8) & 0xFF00) | ((value >> 24) & 0xFF);
}


Eigen::MatrixXd DataIO::readTensor(std::string input_file)
{
    std::ifstream file(input_file, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << input_file << std::endl;
        Eigen::MatrixXd nullMatrix(0,0);
        return nullMatrix;
    }

    // read metadata
    uint32_t magic_number;
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swapEndian(magic_number);

    // Read images
    if (magic_number == 0x00000803)
    {
        // Read the dimensions
        uint32_t num_images, num_rows, num_cols;
        file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
        file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));
        num_images = swapEndian(num_images);
        num_rows = swapEndian(num_rows);
        num_cols = swapEndian(num_cols);

        // Read and fix normalized pixels in a 3D Tensor
        Eigen::MatrixXd imagesTensor(num_images, num_rows * num_cols);

        for (uint32_t i = 0; i < num_images; ++i)
        {
            for (uint32_t j = 0; j < num_rows * num_cols; ++j)
            {
                uint8_t pixel_value;
                file.read(reinterpret_cast<char *>(&pixel_value), sizeof(pixel_value));
                imagesTensor(i, j) = pixel_value / 255.0; // normalize to [0.0, 1.0]
            }
        }

        file.close();
        return imagesTensor;
    }


    // Read labels
    if (magic_number == 0x00000801)
    {
        // Read the number of labels
        uint32_t num_labels;
        file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
        num_labels = swapEndian(num_labels);

        // Read and fix labels in 1D Tensor
        Eigen::MatrixXd labelsTensor(1, num_labels);

        for (uint32_t i = 0; i < num_labels; ++i)
        {
            uint8_t label_value;
            file.read(reinterpret_cast<char *>(&label_value), sizeof(label_value));
            labelsTensor(i) = label_value;
        }

        file.close();
        return labelsTensor;
    }

    file.close();
    std::cerr << "Invalid source data" << std::endl;
    Eigen::MatrixXd nullMatrix(0,0);
    return nullMatrix;
}

void DataIO::writeTensorToFile(const Eigen::MatrixXd &tensor, std::string output_file, int required_element_index)
{
    int image_size = 28 * 28;
    int row_size = 28;
    
    std::ofstream outputFile(output_file);
    if (outputFile.is_open())
    {
        // Write image
        if(tensor.rows() > 1){
            Eigen::VectorXd image_pixels = tensor.row(required_element_index); // Select the row corresponding to the image

            outputFile << 2 << std::endl;
            outputFile << row_size << std::endl;
            outputFile << row_size << std::endl;
            // Writing the pixels of the image on separate rows in the file
            for (uint32_t i = 0; i < image_size; ++i) {
                outputFile << image_pixels(i) << std::endl;
            }
        }
        
        // Write label
        if(tensor.rows() == 1){
            outputFile << 1 << std::endl;
            outputFile << 10 << std::endl;
            
            // Writing 1 where is the number
            for (uint32_t i = 0; i < 10; ++i) {
                if(tensor(required_element_index) == i)
                    outputFile << 1 << std::endl;
                else
                    outputFile << 0 << std::endl;
            }
        }
    }
    else
    {
        std::cerr << "File error" << std::endl;
    }

    outputFile.close();
    std::cout << "The tensor was writen in " << output_file << std::endl;

}