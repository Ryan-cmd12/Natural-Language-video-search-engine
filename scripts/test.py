from src.query.spatial_relationship_filter import (
    SpatialRelationshipFilter,
)


relationship_filter = (
    SpatialRelationshipFilter()
)

result = (
    relationship_filter.evaluate(
        subject_track=car_track,
        object_track=bus_track,
        relationship="left_of",
    )
)

print(result)