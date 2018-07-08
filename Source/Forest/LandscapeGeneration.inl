#pragma once

// Same as the old program, except that it outputs the debug to UE_LOG
class ue_compute_program : public boost::compute::program
{
public:
	void build(const std::string &options = std::string())
	{
		const char *options_string = 0;

		if (!options.empty()) {
			options_string = options.c_str();
		}

		cl_int ret = clBuildProgram(get(), 0, 0, options_string, 0, 0);

#ifdef BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
		if (ret != CL_SUCCESS) {
			std::stringstream errorstringstream;
			// print the error, source code and build log
			errorstringstream << "Boost.Compute: "
				<< "kernel compilation failed (" << ret << ")\n"
				<< "--- source ---\n"
				<< source()
				<< "\n--- build log ---\n"
				<< build_log()
				<< std::endl;

			std::string errorstring(errorstringstream.str());

			UE_LOG(LogTemp, Warning, TEXT("%s"), ANSI_TO_TCHAR(errorstring.c_str()));
		}
#endif

		//if (ret != CL_SUCCESS) {
		//	BOOST_THROW_EXCEPTION(opencl_error(ret));
		//}
	}
};

#include <algorithm>

static std::string read_source_file(const std::string &file)
{
	// open file stream
	std::ifstream stream(file.c_str());

	if (stream.fail()) {
		BOOST_THROW_EXCEPTION(std::ios_base::failure("failed to create stream."));
	}

	// read source
	return std::string(
		(std::istreambuf_iterator<char>(stream)),
		std::istreambuf_iterator<char>()
	);
}

template<class TContainer>
bool begins_with(const TContainer& input, const TContainer& match)
{
	return input.size() >= match.size()
		&& equal(match.begin(), match.end(), input.begin());
}

// Gotta remove unicode BOMs from opencl source files because visual studio 
// is fucking stupid and throws in BOMs in UTF-8 files for no reason
static std::string FixUnicodeBOM(const std::string& inString)
{
	if (inString.length() >= 3 // first 3 bytes of UTF-8 files could be BOM
		&& begins_with(inString, std::string({ '\xEF', '\xBB', '\xBF' })))
	{
		return inString.substr(3);
	}
	else return inString;
}

static boost::compute::program create_with_source_file(const std::string &file,
	const boost::compute::context &context)
{
	const auto source = FixUnicodeBOM(read_source_file(file));
	// create program
	return boost::compute::program::create_with_source(source, context);
}

/// Creates a new program with \p files in \p context.
///
/// \see_opencl_ref{clCreateProgramWithSource}
static boost::compute::program create_with_source_file(const std::vector<std::string> &files,
	const boost::compute::context &context)
{
	std::vector<std::string> sources(files.size());

	for (size_t i = 0; i < files.size(); ++i) {
		// open file stream
		std::ifstream stream(files[i].c_str());

		if (stream.fail()) {
			BOOST_THROW_EXCEPTION(std::ios_base::failure("failed to create stream."));
		}

		// read source
		sources[i] = std::string(
			(std::istreambuf_iterator<char>(stream)),
			std::istreambuf_iterator<char>()
		);

		sources[i] = FixUnicodeBOM(sources[i]);
	}

	// create program
	return boost::compute::program::create_with_source(sources, context);
}