from md_reheader.data.batching import LengthBucketSampler


class TestLengthBucketSampler:
    def test_groups_similar_lengths(self):
        lengths = [500, 600, 8000, 9000, 550, 8500]
        sampler = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192, 32768],
            batch_sizes=[4, 2, 1],
        )
        batches = list(sampler)
        for batch_indices in batches:
            batch_lengths = [lengths[i] for i in batch_indices]
            spread = max(batch_lengths) / min(batch_lengths)
            assert spread < 20

    def test_respects_batch_sizes(self):
        lengths = [100] * 20
        sampler = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192],
            batch_sizes=[4, 2],
            shuffle=False,
        )
        batches = list(sampler)
        for batch in batches:
            assert len(batch) <= 4

    def test_len(self):
        lengths = [100] * 10 + [5000] * 5
        sampler = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192],
            batch_sizes=[4, 2],
        )
        assert len(sampler) == 3 + 3  # ceil(10/4) + ceil(5/2)

    def test_deterministic_with_seed(self):
        lengths = [100, 200, 300, 5000, 6000, 7000]
        s1 = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192],
            batch_sizes=[2, 2],
            seed=42,
        )
        s2 = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192],
            batch_sizes=[2, 2],
            seed=42,
        )
        assert list(s1) == list(s2)

    def test_empty_buckets(self):
        lengths = [100, 200, 300]
        sampler = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192, 32768],
            batch_sizes=[4, 2, 1],
        )
        batches = list(sampler)
        total_items = sum(len(b) for b in batches)
        assert total_items == 3

    def test_all_items_present(self):
        lengths = [100, 5000, 200, 10000, 300]
        sampler = LengthBucketSampler(
            lengths=lengths,
            bucket_boundaries=[2048, 8192, 32768],
            batch_sizes=[4, 2, 1],
            shuffle=False,
        )
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == [0, 1, 2, 3, 4]
