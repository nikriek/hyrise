#pragma once

#include <cxxopts.hpp>
#include <json.hpp>

#include <chrono>
#include <iostream>
#include <unordered_map>

#include "storage/chunk.hpp"
#include "storage/encoding_type.hpp"

namespace opossum {

/**
 * IndividualQueries runs each query a number of times and then the next one
 * PermutedQuerySets runs the queries as sets permuting their order after each run (this exercises caches)
 */
enum class BenchmarkMode { IndividualQueries, PermutedQuerySets };

using Duration = std::chrono::high_resolution_clock::duration;
using TimePoint = std::chrono::high_resolution_clock::time_point;

using NamedQuery = std::pair<std::string, std::string>;
using NamedQueries = std::vector<NamedQuery>;

/**
 * @return std::cout if `verbose` is true, otherwise returns a discarding stream
 */
std::ostream& get_out_stream(const bool verbose);

struct QueryBenchmarkResult {
  size_t num_iterations = 0;
  Duration duration = Duration{};
};

using QueryID = size_t;
using BenchmarkResults = std::unordered_map<std::string, QueryBenchmarkResult>;

/**
 * Loosely copying the functionality of benchmark::State
 * keep_running() returns false once enough iterations or time has passed.
 */
struct BenchmarkState {
  enum class State { NotStarted, Running, Over };

  BenchmarkState(const size_t max_num_iterations, const Duration max_duration);

  bool keep_running();

  State state{State::NotStarted};
  TimePoint begin = TimePoint{};
  TimePoint end = TimePoint{};

  size_t num_iterations = 0;
  size_t max_num_iterations;
  Duration max_duration;
};

struct BenchmarkConfig {
  BenchmarkConfig(const BenchmarkMode benchmark_mode, const bool verbose, const ChunkOffset chunk_size,
                  const EncodingType encoding_type, const size_t max_num_query_runs, const Duration& max_duration,
                  const UseMvcc use_mvcc, const std::optional<std::string>& output_file_path,
                  const bool enable_scheduler, const bool enable_visualization, std::ostream& out);

  static BenchmarkConfig get_default_config();

  const BenchmarkMode benchmark_mode = BenchmarkMode::IndividualQueries;
  const bool verbose = false;
  const ChunkOffset chunk_size = Chunk::MAX_SIZE;
  const EncodingType encoding_type = EncodingType::Dictionary;
  const size_t max_num_query_runs = 1000;
  const Duration max_duration = std::chrono::seconds(5);
  const UseMvcc use_mvcc = UseMvcc::No;
  const std::optional<std::string> output_file_path = std::nullopt;
  const bool enable_scheduler = false;
  const bool enable_visualization = false;
  std::ostream& out;

 private:
  BenchmarkConfig() : out(std::cout) {}
};

class CLIConfigParser {
 public:
  static bool cli_has_json_config(const int argc, char** argv);

  static nlohmann::json parse_json_config_file(const std::string& json_file_str);

  static nlohmann::json basic_cli_options_to_json(const cxxopts::ParseResult& parse_result);

  static BenchmarkConfig parse_basic_options_json_config(const nlohmann::json& json_config);

  static BenchmarkConfig parse_basic_cli_options(const cxxopts::ParseResult& parse_result);
};

}  // namespace opossum
