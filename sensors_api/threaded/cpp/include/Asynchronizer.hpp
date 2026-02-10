#ifndef _ASYNCHRONIZER_HPP
#define _ASYNCHRONIZER_HPP

#include <sl/Camera.hpp>

#include <deque>
#include <mutex>
#include <unordered_map>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>
#include <variant>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <limits>
#include <iostream>

// =============================================================================
// Data Packets
// =============================================================================

struct ZedDataPacket {
    sl::Mat image;
    sl::Mat pointCloud;
    sl::Objects objects;
    sl::Bodies bodies;
    sl::Pose pose;
    sl::Timestamp timestamp;
    bool valid = false;
    bool hasObjects = false;
    bool hasBodies = false;
    bool hasTrackingPose = false;
    sl::Transform sensorPose;
    bool hasSensorPose = false;
};

struct LidarDataPacket {
    sl::Mat pointCloud;
    sl::Mat intensityImage;
    sl::Timestamp timestamp;
    bool valid = false;
    sl::Transform sensorPose;
    bool hasPose = false;
};

// =============================================================================
// Synchronizer types
// =============================================================================

using ZedPacketPtr = std::shared_ptr<ZedDataPacket>;
using LidarPacketPtr = std::shared_ptr<LidarDataPacket>;
using PacketVariant = std::variant<ZedPacketPtr, LidarPacketPtr>;

struct TimedPacket {
    int64_t ts_ns;
    PacketVariant packet;
};

struct Bundle {
    int64_t anchor_ts_ns = 0;
    std::string anchor_stream;
    std::unordered_map<std::string, PacketVariant> packets;
    std::vector<std::string> missing;
    std::vector<double> timediff; // aligned with missing (only push alongside missing)
};

enum class MissingPolicy {
    EmitPartial,
    DropIfMissingRequired
};

enum class Role {
    Required,
    Optional
};

// =============================================================================
// Helpers
// =============================================================================

inline sl::Mat deepCopyToGPU(const sl::Mat& src) {
    sl::Mat dst;
    if (src.getWidth() == 0 || src.getHeight() == 0)
        return dst;

    dst.alloc(src.getWidth(), src.getHeight(), src.getDataType(), sl::MEM::GPU);
    if (src.getMemoryType() == sl::MEM::CPU)
        dst.setFrom(src, sl::COPY_TYPE::CPU_GPU);
    else
        dst.setFrom(src, sl::COPY_TYPE::GPU_GPU);
    return dst;
}

inline ZedPacketPtr deepCopyPacketPtr(const ZedDataPacket& p) {
    auto out = std::make_shared<ZedDataPacket>();
    out->image = deepCopyToGPU(p.image);
    out->pointCloud = deepCopyToGPU(p.pointCloud);

    out->objects = p.objects;
    out->bodies = p.bodies;
    out->pose = p.pose;
    out->timestamp = p.timestamp;

    out->valid = p.valid;
    out->hasObjects = p.hasObjects;
    out->hasBodies = p.hasBodies;
    out->hasTrackingPose = p.hasTrackingPose;

    out->sensorPose = p.sensorPose;
    out->hasSensorPose = p.hasSensorPose;
    return out;
}

inline LidarPacketPtr deepCopyPacketPtr(const LidarDataPacket& p) {
    auto out = std::make_shared<LidarDataPacket>();
    out->pointCloud = deepCopyToGPU(p.pointCloud);
    out->intensityImage = deepCopyToGPU(p.intensityImage);
    out->timestamp = p.timestamp;
    out->valid = p.valid;
    out->sensorPose = p.sensorPose;
    out->hasPose = p.hasPose;
    return out;
}

inline int64_t packetTimestampNs(const PacketVariant& v) {
    return std::visit(
        [](const auto& ptr) -> int64_t {
            return static_cast<int64_t>(ptr->timestamp.getNanoseconds());
        },
        v
    );
}

// =============================================================================
// AsyncPacketSynchronizer
// =============================================================================

class AsyncPacketSynchronizer {
public:
    AsyncPacketSynchronizer(int64_t tolerance_ns, size_t max_buffer_per_stream = 60)
        : tolerance_ns_(tolerance_ns)
        , max_buffer_(max_buffer_per_stream) { }

    // ---------------- Config ----------------
    void setToleranceNs(int64_t t) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        tolerance_ns_ = t;
    }

    void setMaxBufferPerStream(size_t n) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        max_buffer_ = n;
    }

    void setAnchor(const std::string& stream) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        anchor_ = stream;
    }

    // ---------------- Roles ----------------
    void setStreamRole(const std::string& stream, Role role) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        roles_[stream] = role;
    }

    Role getStreamRole(const std::string& stream) const {
        std::lock_guard<std::mutex> lk(map_mtx_);
        auto it = roles_.find(stream);
        return (it == roles_.end()) ? Role::Required : it->second;
    }

    // ---------------- Debug skew histograms ----------------
    void enableDebug(bool on, int64_t bin_width_ns = 100'000, int bins = 201) {
        debug_enabled_.store(on);
        if (!on)
            return;
        std::lock_guard<std::mutex> lk(debug_mtx_);
        debug_bin_width_ns_ = bin_width_ns;
        debug_bins_ = bins;
        for (auto& [_, h] : skew_)
            h = SkewHistogram(bin_width_ns, bins);
    }

    void resetDebug() {
        std::lock_guard<std::mutex> lk(debug_mtx_);
        for (auto& [_, h] : skew_)
            h.reset();
    }

    void printSkewSummary(std::ostream& os) const {
        std::lock_guard<std::mutex> lk(debug_mtx_);
        for (const auto& [stream, h] : skew_) {
            os << "Skew[" << stream << "] n=" << h.n << " mean(ms)=" << h.mean_ns() / 1e6 << " std(ms)=" << h.stddev_ns() / 1e6
               << " min(ms)=" << (h.n ? (double)h.min_ns / 1e6 : 0.0) << " max(ms)=" << (h.n ? (double)h.max_ns / 1e6 : 0.0) << "\n";
        }
    }

    // ---------------- Ingest ----------------
    void ingest(const std::string& stream, const ZedDataPacket& pkt) {
        ingestImpl(stream, PacketVariant {deepCopyPacketPtr(pkt)});
    }

    void ingest(const std::string& stream, const LidarDataPacket& pkt) {
        ingestImpl(stream, PacketVariant {deepCopyPacketPtr(pkt)});
    }

    bool removeStream(const std::string& stream) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        auto it = streams_.find(stream);
        if (it == streams_.end())
            return false;

        streams_.erase(it);
        roles_.erase(stream);

        if (anchor_ == stream) {
            anchor_.clear();
            if (!streams_.empty())
                anchor_ = streams_.begin()->first;
        }
        return true;
    }

    std::vector<std::string> listStreams() const {
        std::lock_guard<std::mutex> lk(map_mtx_);
        std::vector<std::string> out;
        out.reserve(streams_.size());
        for (const auto& kv : streams_)
            out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    void printLastSyncDebug(std::ostream& os) const {
        std::lock_guard<std::mutex> lk(last_dbg_mtx_);

        if (!last_dbg_.emitted_bundle) {
            os << "[syncNext debug] No bundle emitted yet.\n";
            return;
        }

        os << "---- syncNext debug ----\n";
        os << "seq=" << last_dbg_.seq << " tol(ms)=" << (double)last_dbg_.tolerance_ns / 1e6 << " anchor=" << last_dbg_.anchor_stream
           << " anchor_ts(ms)=" << (double)last_dbg_.anchor_ts_ns / 1e6 << " anchor_idx=" << last_dbg_.anchor_taken_index
           << " policy=" << (last_dbg_.policy == MissingPolicy::EmitPartial ? "EmitPartial" : "DropIfMissingRequired") << "\n";

        os << "targets:";
        for (auto& t : last_dbg_.targets_sorted)
            os << " " << t;
        os << "\n";

        // Print per stream in a stable order
        auto order = last_dbg_.targets_sorted;
        for (const auto& s : order) {
            auto it = last_dbg_.per_stream.find(s);
            if (it == last_dbg_.per_stream.end())
                continue;
            const auto& tr = it->second;

            os << "  [" << s << "]" << (tr.used_as_anchor ? " (ANCHOR)" : "") << " q_before=" << tr.q_size_before
               << " pruned=" << tr.pruned_front << " q_after=" << tr.q_size_after;

            if (tr.matched) {
                os << " MATCH idx=" << tr.taken_index << " ts(ms)=" << (double)tr.taken_ts_ns / 1e6
                   << " dt(ms)=" << (double)tr.dt_signed_ns / 1e6 << " |dt|(ms)=" << (double)tr.dt_abs_ns / 1e6;
            } else if (tr.missing) {
                os << " MISSING";
                if (!tr.note.empty())
                    os << " (" << tr.note << ")";
            } else {
                os << " (no decision)";
            }

            os << "\n";
        }

        os << "------------------------\n";
    }

    std::optional<Bundle>
    syncNext(MissingPolicy policy = MissingPolicy::EmitPartial, const std::vector<std::string>& requested_streams = {}) {
        int64_t tol;
        std::vector<std::string> targets;
        std::unordered_map<std::string, std::shared_ptr<StreamState>> states;
        std::unordered_map<std::string, Role> roles_snapshot;

        // Prepare last sync debug (we'll fill it as we go)
        LastSyncDebug local_dbg;
        local_dbg.clear();

        // ---------------- Snapshot config / targets / states / roles ----------------
        {
            std::lock_guard<std::mutex> lk(map_mtx_);
            tol = tolerance_ns_;

            local_dbg.seq = ++sync_seq_;
            local_dbg.tolerance_ns = tol;
            local_dbg.policy = policy;

            if (!requested_streams.empty()) {
                targets = requested_streams;
            } else {
                targets.reserve(streams_.size());
                for (const auto& [name, _] : streams_)
                    targets.push_back(name);
            }

            // Remove duplicates just in case
            std::sort(targets.begin(), targets.end());
            targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

            // Snapshot stream states + roles for these targets
            for (const auto& name : targets) {
                auto it = streams_.find(name);
                if (it != streams_.end() && it->second) {
                    states[name] = it->second;
                }
                auto rt = roles_.find(name);
                roles_snapshot[name] = (rt == roles_.end()) ? Role::Required : rt->second;
            }
        }

        if (targets.empty())
            return std::nullopt;

        // We'll show debug for all requested targets (even if missing in states)
        local_dbg.targets_sorted = targets; // already sorted above
        for (const auto& s : local_dbg.targets_sorted) {
            local_dbg.per_stream[s].present_in_targets = true;
        }

        // If nothing exists, no bundle
        if (states.empty()) {
            for (const auto& s : targets) {
                auto& tr = local_dbg.per_stream[s];
                tr.missing = true;
                tr.note = "stream not present";
                tr.q_size_before = 0;
                tr.q_size_after = 0;
            }
            // publish debug (optional)
            {
                std::lock_guard<std::mutex> lk(last_dbg_mtx_);
                last_dbg_ = std::move(local_dbg);
            }
            return std::nullopt;
        }

        // ---------------- Lock streams in deterministic order ----------------
        std::vector<std::string> lock_order;
        lock_order.reserve(states.size());
        for (const auto& [name, _] : states)
            lock_order.push_back(name);
        std::sort(lock_order.begin(), lock_order.end());

        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(lock_order.size());
        for (const auto& s : lock_order) {
            auto it = states.find(s);
            if (it != states.end() && it->second) {
                locks.emplace_back(it->second->mtx);
            }
        }

        // Record q_size_before now that locks are held
        for (const auto& s : lock_order) {
            auto it = states.find(s);
            if (it == states.end() || !it->second)
                continue;
            local_dbg.per_stream[s].q_size_before = it->second->q.size();
        }

        // ---------------- Helpers ----------------
        auto lowerBoundIndex = [](const std::deque<TimedPacket>& dq, int64_t key_ns) -> size_t {
            auto it = std::lower_bound(dq.begin(), dq.end(), key_ns, [](const TimedPacket& a, int64_t ts) {
                return a.ts_ns < ts;
            });
            return (size_t)std::distance(dq.begin(), it);
        };

        // Closest index to target_ns, BUT only considering elements with ts >= (target_ns - tol)
        // This mirrors your prune rule and fixes "scores good but then empty after prune".
        auto closestIndexRespectingPrune = [&](const std::deque<TimedPacket>& dq, int64_t target_ns) -> std::optional<size_t> {
            if (dq.empty())
                return std::nullopt;
            const int64_t min_ok = target_ns - tol;

            size_t start = lowerBoundIndex(dq, min_ok);
            if (start >= dq.size())
                return std::nullopt; // everything too old (would be pruned)

            // Now find insertion point for target_ns, but clamp to [start, end]
            size_t ins = lowerBoundIndex(dq, target_ns);
            if (ins < start)
                ins = start;

            if (ins == start)
                return start;
            if (ins >= dq.size())
                return dq.size() - 1;

            // Compare ins vs ins-1, but ins-1 may be < start => clamp
            size_t i1 = ins;
            size_t i0 = (ins > start) ? (ins - 1) : start;

            int64_t d1 = std::llabs(dq[i1].ts_ns - target_ns);
            int64_t d0 = std::llabs(dq[i0].ts_ns - target_ns);
            return (d1 < d0) ? i1 : i0;
        };

        auto hasOptionalData = [&]() -> bool {
            for (const auto& s : targets) {
                auto it = states.find(s);
                if (it == states.end() || !it->second)
                    continue;
                if (roles_snapshot[s] == Role::Optional && !it->second->q.empty())
                    return true;
            }
            return false;
        };

        // Score a candidate anchor timestamp T: how many streams can match within tol?
        // Uses closestIndexRespectingPrune() so "too old" packets don't count.
        auto scoreAnchorTs = [&](int64_t T, const std::string& anchor_stream) -> std::pair<int, int> {
            int required_matches = 0;
            int total_matches = 0;

            for (const auto& s : targets) {
                if (s == anchor_stream)
                    continue;

                auto it = states.find(s);
                if (it == states.end() || !it->second)
                    continue;
                const auto& dq = it->second->q;
                if (dq.empty())
                    continue;

                auto idxOpt = closestIndexRespectingPrune(dq, T);
                if (!idxOpt)
                    continue;

                const size_t idx = *idxOpt;
                const int64_t dt_abs = std::llabs(dq[idx].ts_ns - T);
                if (dt_abs <= tol) {
                    total_matches++;
                    if (roles_snapshot[s] == Role::Required)
                        required_matches++;
                }
            }

            return {required_matches, total_matches};
        };

        // Optional fallback guard:
        // If there exists any OPTIONAL stream in targets, and we have seen OPTIONAL recently,
        // don't let REQUIRED anchors run too far ahead when OPTIONAL queues are empty.
        // This reduces anchor "flapping" and ZED racing ahead when LiDAR is late.
        const bool optional_exists_in_targets = [&]() {
            for (const auto& s : targets)
                if (roles_snapshot[s] == Role::Optional)
                    return true;
            return false;
        }();

        // Update last_optional_seen_ts_ns_ from currently available optional packets
        if (optional_exists_in_targets) {
            int64_t best_seen = last_optional_seen_ts_ns_.load();
            for (const auto& s : targets) {
                if (roles_snapshot[s] != Role::Optional)
                    continue;
                auto it = states.find(s);
                if (it == states.end() || !it->second)
                    continue;
                const auto& q = it->second->q;
                if (!q.empty())
                    best_seen = std::max(best_seen, q.back().ts_ns);
            }
            last_optional_seen_ts_ns_.store(best_seen);
        }

        const bool optional_has_data_now = optional_exists_in_targets ? hasOptionalData() : false;

        // Pick best anchor packet among all available streams:
        // - Consider a few packets from each stream (depth)
        // - Prefer higher required_matches, then higher total_matches
        // - Prefer OPTIONAL (LiDAR) on ties
        // - Prefer earlier timestamp (less latency) on ties
        auto pick_anchor_scored = [&]() -> std::optional<std::pair<std::string, size_t>> {
            const size_t depth = std::max<size_t>(1, anchor_search_depth_);

            bool found = false;
            std::string best_stream;
            size_t best_idx = 0;
            int best_req = -1;
            int best_total = -1;
            int best_role_pref = -1; // OPTIONAL preferred => 1, else 0
            int64_t best_T = 0;

            // Fallback constraint when OPTIONAL exists but currently has no data
            const int64_t last_opt_seen = last_optional_seen_ts_ns_.load();
            const bool apply_fallback_guard
                = optional_exists_in_targets && !optional_has_data_now && (last_opt_seen != std::numeric_limits<int64_t>::min());

            for (const auto& s : lock_order) {
                auto it = states.find(s);
                if (it == states.end() || !it->second)
                    continue;
                const auto& q = it->second->q;
                if (q.empty())
                    continue;

                const size_t max_idx = std::min(depth - 1, q.size() - 1);

                for (size_t i = 0; i <= max_idx; ++i) {
                    const int64_t T = q[i].ts_ns;

                    // If OPTIONAL exists but is empty right now, don't let REQUIRED lead too far ahead
                    if (apply_fallback_guard && roles_snapshot[s] == Role::Required) {
                        if (T > last_opt_seen + zed_fallback_max_lead_ns_) {
                            continue;
                        }
                    }

                    auto [req_m, tot_m] = scoreAnchorTs(T, s);

                    // Gate: require at least one match besides anchor to avoid emitting lonely anchors
                    // Exception: allow single-stream scenarios (e.g., LiDAR-only mode)
                    if (tot_m == 0 && lock_order.size() > 1)
                        continue;

                    const int role_pref = (roles_snapshot[s] == Role::Optional) ? 1 : 0;

                    const bool better = (!found) || (req_m > best_req) || (req_m == best_req && tot_m > best_total)
                        || (req_m == best_req && tot_m == best_total && role_pref > best_role_pref)
                        || (req_m == best_req && tot_m == best_total && role_pref == best_role_pref && T < best_T);

                    if (better) {
                        found = true;
                        best_stream = s;
                        best_idx = i;
                        best_req = req_m;
                        best_total = tot_m;
                        best_role_pref = role_pref;
                        best_T = T;
                    }
                }
            }

            if (!found)
                return std::nullopt;
            return std::make_pair(best_stream, best_idx);
        };

        auto best = pick_anchor_scored();
        if (!best) {
            // publish debug that nothing emitted (optional)
            {
                std::lock_guard<std::mutex> lk(last_dbg_mtx_);
                last_dbg_ = std::move(local_dbg);
            }
            return std::nullopt;
        }

        const std::string anchor = best->first;
        const size_t anchor_idx = best->second;

        auto itAnchor = states.find(anchor);
        if (itAnchor == states.end() || !itAnchor->second)
            return std::nullopt;
        auto& anchorQ = itAnchor->second->q;
        if (anchorQ.empty() || anchor_idx >= anchorQ.size())
            return std::nullopt;

        // Debug fields for anchor
        local_dbg.anchor_stream = anchor;
        local_dbg.anchor_ts_ns = anchorQ[anchor_idx].ts_ns;
        local_dbg.anchor_taken_index = anchor_idx;

        {
            auto& atr = local_dbg.per_stream[anchor];
            atr.used_as_anchor = true;
            atr.matched = true;
            atr.taken_index = anchor_idx;
            atr.taken_ts_ns = local_dbg.anchor_ts_ns;
            atr.anchor_ts_ns = local_dbg.anchor_ts_ns;
            atr.dt_signed_ns = 0;
            atr.dt_abs_ns = 0;
        }

        // Consume anchor up to idx (inclusive)
        TimedPacket anchorSample = anchorQ[anchor_idx];
        anchorQ.erase(anchorQ.begin(), anchorQ.begin() + (anchor_idx + 1));

        Bundle b;
        b.anchor_ts_ns = anchorSample.ts_ns;
        b.anchor_stream = anchor;
        b.packets[anchor] = anchorSample.packet;

        // Keep skew histogram alive for anchor too (dt=0)
        recordSkewLocked(anchor, 0);

        bool must_drop = false;

        // ---------------- Build bundle for other targets ----------------
        for (const auto& s : targets) {
            if (s == anchor)
                continue;

            auto& tr = local_dbg.per_stream[s];
            tr.anchor_ts_ns = b.anchor_ts_ns;

            auto it = states.find(s);
            if (it == states.end() || !it->second) {
                tr.missing = true;
                tr.note = "stream not present";
                b.missing.push_back(s);
                b.timediff.push_back(1e30);
                if (policy == MissingPolicy::DropIfMissingRequired && roles_snapshot[s] == Role::Required) {
                    must_drop = true;
                }
                continue;
            }

            auto& dq = it->second->q;

            // Prune too-old relative to anchor time
            const size_t before_prune = dq.size();
            const int64_t too_old = b.anchor_ts_ns - tol;
            while (!dq.empty() && dq.front().ts_ns < too_old)
                dq.pop_front();
            tr.pruned_front = before_prune - dq.size();

            if (dq.empty()) {
                tr.missing = true;
                tr.note = "empty after prune";
                b.missing.push_back(s);
                b.timediff.push_back(1e30);
                if (policy == MissingPolicy::DropIfMissingRequired && roles_snapshot[s] == Role::Required) {
                    must_drop = true;
                }
                continue;
            }

            // Both REQUIRED and OPTIONAL use "closest"
            auto idxOpt = closestIndexRespectingPrune(dq, b.anchor_ts_ns);
            if (!idxOpt) {
                tr.missing = true;
                tr.note = "empty after prune";
                b.missing.push_back(s);
                b.timediff.push_back(1e30);
                if (policy == MissingPolicy::DropIfMissingRequired && roles_snapshot[s] == Role::Required) {
                    must_drop = true;
                }
                continue;
            }

            const size_t idx = *idxOpt;
            const int64_t dt_signed = dq[idx].ts_ns - b.anchor_ts_ns;
            const int64_t dt_abs = std::llabs(dt_signed);

            tr.taken_index = idx;
            tr.taken_ts_ns = dq[idx].ts_ns;
            tr.dt_signed_ns = dt_signed;
            tr.dt_abs_ns = dt_abs;

            if (dt_abs <= tol) {
                b.packets[s] = dq[idx].packet;

                // Consume up to idx inclusive to prevent backlog/latency
                dq.erase(dq.begin(), dq.begin() + (idx + 1));

                tr.matched = true;
                recordSkewLocked(s, dt_signed);
            } else {
                tr.missing = true;
                tr.note = "closest out of tol";
                b.missing.push_back(s);
                b.timediff.push_back((double)dt_abs);
                if (policy == MissingPolicy::DropIfMissingRequired && roles_snapshot[s] == Role::Required) {
                    must_drop = true;
                }
            }
        }

        if (policy == MissingPolicy::DropIfMissingRequired && must_drop) {
            // still publish debug before returning (helpful when debugging why it drops)
            for (const auto& s : lock_order) {
                auto it = states.find(s);
                if (it == states.end() || !it->second)
                    continue;
                local_dbg.per_stream[s].q_size_after = it->second->q.size();
            }
            local_dbg.emitted_bundle = false;
            {
                std::lock_guard<std::mutex> lk(last_dbg_mtx_);
                last_dbg_ = std::move(local_dbg);
            }
            return std::nullopt;
        }

        // Record q_size_after now that we've consumed/pruned
        for (const auto& s : lock_order) {
            auto it = states.find(s);
            if (it == states.end() || !it->second)
                continue;
            local_dbg.per_stream[s].q_size_after = it->second->q.size();
        }

        local_dbg.emitted_bundle = true;

        // publish last debug
        {
            std::lock_guard<std::mutex> lk(last_dbg_mtx_);
            last_dbg_ = std::move(local_dbg);
        }

        return b;
    }

private:
    // ---------------- Stream state ----------------
    struct StreamState {
        mutable std::mutex mtx;
        std::deque<TimedPacket> q;
    };

    void ingestImpl(const std::string& stream, PacketVariant&& pkt) {
        std::shared_ptr<StreamState> st;

        {
            std::lock_guard<std::mutex> lk(map_mtx_);
            if (anchor_.empty())
                anchor_ = stream;

            auto& ref = streams_[stream];
            if (!ref)
                ref = std::make_shared<StreamState>();
            st = ref;

            // Default role: Required unless user overrides later
            if (roles_.find(stream) == roles_.end())
                roles_[stream] = Role::Required;
        }

        const int64_t ts_ns = packetTimestampNs(pkt);

        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->q.push_back(TimedPacket {ts_ns, std::move(pkt)});
            while (st->q.size() > max_buffer_)
                st->q.pop_front();
        }
    }

    // ---------------- Debug histogram ----------------
    struct SkewHistogram {
        int64_t bin_width_ns;
        int bins;
        std::vector<uint64_t> counts;

        uint64_t n = 0;
        int64_t min_ns = std::numeric_limits<int64_t>::max();
        int64_t max_ns = std::numeric_limits<int64_t>::min();
        long double sum = 0;
        long double sumsq = 0;

        SkewHistogram(int64_t bw = 100'000, int b = 201)
            : bin_width_ns(bw)
            , bins(b)
            , counts((size_t)b, 0) { }

        void reset() {
            std::fill(counts.begin(), counts.end(), 0);
            n = 0;
            min_ns = std::numeric_limits<int64_t>::max();
            max_ns = std::numeric_limits<int64_t>::min();
            sum = 0;
            sumsq = 0;
        }

        void add(int64_t dt_ns) {
            n++;
            min_ns = std::min(min_ns, dt_ns);
            max_ns = std::max(max_ns, dt_ns);
            sum += (long double)dt_ns;
            sumsq += (long double)dt_ns * (long double)dt_ns;

            const int center = bins / 2;
            long long idx = center + (long long)std::llround((long double)dt_ns / (long double)bin_width_ns);
            if (idx < 0)
                idx = 0;
            if (idx >= bins)
                idx = bins - 1;
            counts[(size_t)idx]++;
        }

        double mean_ns() const {
            return n ? (double)(sum / (long double)n) : 0.0;
        }
        double stddev_ns() const {
            if (n < 2)
                return 0.0;
            long double mean = sum / (long double)n;
            long double var = (sumsq / (long double)n) - mean * mean;
            if (var < 0)
                var = 0;
            return std::sqrt((double)var);
        }
    };

    void recordSkewLocked(const std::string& stream, int64_t dt_ns) {
        if (!debug_enabled_.load())
            return;
        std::lock_guard<std::mutex> lk(debug_mtx_);
        auto it = skew_.find(stream);
        if (it == skew_.end()) {
            skew_.emplace(stream, SkewHistogram(debug_bin_width_ns_, debug_bins_));
        }
        skew_[stream].add(dt_ns);
    }

    // ---------------- Per-sync debug trace ----------------
    struct SyncStreamTrace {
        bool present_in_targets = false;

        // Queue bookkeeping
        size_t q_size_before = 0;
        size_t q_size_after = 0;
        size_t pruned_front = 0;

        // Selection info
        bool used_as_anchor = false;
        bool matched = false;
        bool missing = false;

        // Indices into the queue as it existed at decision time
        // (before any erase/pop for this stream in this sync)
        size_t taken_index = (size_t)-1;

        // Timestamps + diffs
        int64_t taken_ts_ns = 0;
        int64_t anchor_ts_ns = 0;
        int64_t dt_signed_ns = 0;
        int64_t dt_abs_ns = 0;

        // Why it failed (optional)
        std::string note;
    };

    struct LastSyncDebug {
        uint64_t seq = 0;
        int64_t tolerance_ns = 0;

        std::string anchor_stream;
        int64_t anchor_ts_ns = 0;
        size_t anchor_taken_index = (size_t)-1;

        std::vector<std::string> targets_sorted;
        std::unordered_map<std::string, SyncStreamTrace> per_stream;

        bool emitted_bundle = false;
        MissingPolicy policy = MissingPolicy::EmitPartial;

        void clear() {
            emitted_bundle = false;
            anchor_stream.clear();
            anchor_ts_ns = 0;
            anchor_taken_index = (size_t)-1;
            targets_sorted.clear();
            per_stream.clear();
        }
    };

    mutable std::mutex last_dbg_mtx_;
    LastSyncDebug last_dbg_;
    std::atomic<uint64_t> sync_seq_ {0};

private:
    // Map + config + roles
    mutable std::mutex map_mtx_;
    std::unordered_map<std::string, std::shared_ptr<StreamState>> streams_;
    std::unordered_map<std::string, Role> roles_;

    std::string anchor_;
    int64_t tolerance_ns_;
    size_t max_buffer_;
    size_t anchor_search_depth_ = 5;

    // Debug
    std::atomic<bool> debug_enabled_ {false};
    mutable std::mutex debug_mtx_;
    std::unordered_map<std::string, SkewHistogram> skew_;
    int64_t debug_bin_width_ns_ = 100'000;
    int debug_bins_ = 201;

    // --- ZED fallback guard when OPTIONAL (LiDAR) exists but is temporarily empty ---
    std::atomic<int64_t> last_optional_seen_ts_ns_ {std::numeric_limits<int64_t>::min()};
    // How far ZED is allowed to run ahead of last seen LiDAR timestamp when LiDAR queue is empty.
    int64_t zed_fallback_max_lead_ns_ = 100'000'000; // default 100ms for 10Hz LiDAR mode
};

#endif
