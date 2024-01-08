from django.urls import include, path
from django.contrib import admin
from rest_framework import routers
from api.user_management.views import UserViewSet, GroupViewSet
from api.analysis.views import AnalysisView
from api.analysis.views import TaskStatusView
from api.analysis.views import ModelsView

router = routers.DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'groups', GroupViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('analysis/', AnalysisView.as_view()),
    path('task-status/<str:task_id>/', TaskStatusView.as_view(), name='task-status'),
    path('analysis/models', ModelsView.as_view()),
]