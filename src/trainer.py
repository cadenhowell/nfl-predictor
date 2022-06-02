from dataloader import ColligatePotentialDataset


def main():
    include_combine=True
    data = ColligatePotentialDataset('rushing', include_combine)
    print(len(data.madden_stats))
    print(len(data.college_stats.index.unique()))
    print(len(data.schools))
    if include_combine:
        print(len(data.combine_stats))
    print(data[179])

if __name__ == '__main__':
    main()