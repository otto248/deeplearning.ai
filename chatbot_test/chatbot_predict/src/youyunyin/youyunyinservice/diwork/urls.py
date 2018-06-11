from django.conf.urls import url

from . import views

#router = DefaultRouter()
#router.register('dirty_to_mongo',views.dirty_to_mongo, base_name='dirty_to_mongo')

urlpatterns = [
    url('extract_origin/', views.extract_origin),
    url('to_database/',views.to_database),
    url('training_classifier/',views.training_classifier),
    url('training_w2v/', views.training_w2v),
]
